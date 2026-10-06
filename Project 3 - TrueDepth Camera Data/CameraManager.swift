import AVFoundation
import Combine
import CoreImage
import UIKit

final class CameraManager: NSObject, ObservableObject, AVCaptureDataOutputSynchronizerDelegate {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private let queue = DispatchQueue(label: "capture.queue")
    private var latestDepth: AVDepthData?
    private var latestPixelBuffer: CVPixelBuffer?
    private var frameCount = 0

    @Published var status = "Starting..."
    @Published var saveMessage = ""
    @Published var depthImage: UIImage?
    @Published var savedFiles: [URL] = []

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else {
                DispatchQueue.main.async { self.status = "Camera permission denied" }
                return
            }
            self.queue.async {
                self.configure()
                self.session.startRunning()
            }
        }
    }

    private func width(_ f: AVCaptureDevice.Format) -> Int32 {
        CMVideoFormatDescriptionGetDimensions(f.formatDescription).width
    }

    private func configure() {
        guard let device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device) else {
            DispatchQueue.main.async { self.status = "No TrueDepth camera found" }
            return
        }

        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        session.addInput(input)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        session.addOutput(videoOutput)
        session.addOutput(depthOutput)
        depthOutput.isFilteringEnabled = false   // raw, unsmoothed depth

        do {
            try device.lockForConfiguration()
            let candidates = device.formats.filter { !$0.supportedDepthDataFormats.isEmpty }
            if let best = candidates.max(by: { width($0) < width($1) }) {
                device.activeFormat = best
                let depthFormats = best.supportedDepthDataFormats.filter {
                    let t = CMFormatDescriptionGetMediaSubType($0.formatDescription)
                    return t == kCVPixelFormatType_DepthFloat32 || t == kCVPixelFormatType_DepthFloat16
                }
                if let d = depthFormats.max(by: { width($0) < width($1) }) {
                    device.activeDepthDataFormat = d
                }
            }
            device.unlockForConfiguration()
        } catch {
            DispatchQueue.main.async { self.status = "Config error: \(error.localizedDescription)" }
        }

        synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
        synchronizer?.setDelegate(self, queue: queue)
        session.commitConfiguration()
    }

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let syncedDepth = collection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData,
              let syncedVideo = collection.synchronizedData(for: videoOutput) as? AVCaptureSynchronizedSampleBufferData,
              !syncedDepth.depthDataWasDropped, !syncedVideo.sampleBufferWasDropped else { return }

        var depth = syncedDepth.depthData
        if depth.depthDataType != kCVPixelFormatType_DepthFloat32 {
            depth = depth.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        }
        latestDepth = depth
        latestPixelBuffer = CMSampleBufferGetImageBuffer(syncedVideo.sampleBuffer)

        let map = depth.depthDataMap
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        CVPixelBufferLockBaseAddress(map, .readOnly)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        let row = CVPixelBufferGetBaseAddress(map)!.advanced(by: (h / 2) * rowBytes)
        let center = row.assumingMemoryBound(to: Float32.self)[w / 2]
        CVPixelBufferUnlockBaseAddress(map, .readOnly)

        let text = String(format: "Depth %dx%d | center %.0f mm | filtered: %@",
                          w, h, center * 1000, depth.isDepthDataFiltered ? "yes" : "no")

        frameCount += 1
        let image = (frameCount % 3 == 0) ? makeDepthImage(map) : nil
        DispatchQueue.main.async {
            self.status = text
            if let image = image { self.depthImage = image }
        }
    }

    // Grayscale depth view: near = bright, far = dark (0.15 m to 0.8 m)
    private func makeDepthImage(_ map: CVPixelBuffer) -> UIImage? {
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        let base = CVPixelBufferGetBaseAddress(map)!
        var pixels = [UInt8](repeating: 0, count: w * h)
        let nearM: Float = 0.15, farM: Float = 0.8
        for y in 0..<h {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
            for x in 0..<w {
                let d = row[x]
                if d.isFinite && d > 0 {
                    let t = 1 - min(max((d - nearM) / (farM - nearM), 0), 1)
                    pixels[y * w + x] = UInt8(t * 255)
                }
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8,
                               bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false,
                               intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cg, scale: 1, orientation: .leftMirrored)
    }

    func saveFrame() {
        queue.async {
            guard let depth = self.latestDepth, let pb = self.latestPixelBuffer else {
                DispatchQueue.main.async { self.saveMessage = "No frame yet" }
                return
            }
            do {
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let dir = docs.appendingPathComponent("capture_\(Int(Date().timeIntervalSince1970))")
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

                // RGB
                let ctx = CIContext()
                guard let png = ctx.pngRepresentation(of: CIImage(cvPixelBuffer: pb), format: .RGBA8,
                                                      colorSpace: CGColorSpaceCreateDeviceRGB()) else {
                    throw NSError(domain: "save", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "RGB encode failed"])
                }
                let rgbURL = dir.appendingPathComponent("rgb.png")
                try png.write(to: rgbURL)

                // Depth: raw float32 meters, row-major
                let map = depth.depthDataMap
                let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
                CVPixelBufferLockBaseAddress(map, .readOnly)
                let rowBytes = CVPixelBufferGetBytesPerRow(map)
                let base = CVPixelBufferGetBaseAddress(map)!
                var data = Data()
                for y in 0..<h { data.append(Data(bytes: base + y * rowBytes, count: w * 4)) }
                CVPixelBufferUnlockBaseAddress(map, .readOnly)
                let depthURL = dir.appendingPathComponent("depth_\(w)x\(h)_float32_m.bin")
                try data.write(to: depthURL)

                // Intrinsics + metadata
                var meta: [String: Any] = [
                    "depth_width": w, "depth_height": h,
                    "rgb_width": CVPixelBufferGetWidth(pb), "rgb_height": CVPixelBufferGetHeight(pb),
                    "depth_filtered": depth.isDepthDataFiltered
                ]
                if let cal = depth.cameraCalibrationData {
                    let m = cal.intrinsicMatrix
                    let d = cal.intrinsicMatrixReferenceDimensions
                    meta["fx"] = Double(m.columns.0.x); meta["fy"] = Double(m.columns.1.y)
                    meta["cx"] = Double(m.columns.2.x); meta["cy"] = Double(m.columns.2.y)
                    meta["intrinsics_ref_width"] = Double(d.width)
                    meta["intrinsics_ref_height"] = Double(d.height)
                }
                let json = try JSONSerialization.data(withJSONObject: meta, options: .prettyPrinted)
                let metaURL = dir.appendingPathComponent("meta.json")
                try json.write(to: metaURL)

                DispatchQueue.main.async {
                    self.saveMessage = "Saved \(dir.lastPathComponent)"
                    self.savedFiles = [rgbURL, depthURL, metaURL]
                }
            } catch {
                DispatchQueue.main.async { self.saveMessage = "Save failed: \(error.localizedDescription)" }
            }
        }
    }
}
