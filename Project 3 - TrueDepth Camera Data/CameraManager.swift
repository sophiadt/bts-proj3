import AVFoundation
import Combine
import CoreImage
import MediaPipeTasksVision
import UIKit
import simd

final class CameraManager: NSObject, ObservableObject, AVCaptureDataOutputSynchronizerDelegate {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private let queue = DispatchQueue(label: "capture.queue")
    private var latestDepth: AVDepthData?
    private var latestPixelBuffer: CVPixelBuffer?
    private var frameCount = 0
    private var landmarker: FaceLandmarker?
    private var pending = 0
    private let ciContext = CIContext()

    @Published var status = "Starting..."
    @Published var saveMessage = ""
    @Published var prompt = ""
    @Published var isCapturing = false
    @Published var depthImage: UIImage?
    @Published var savedFiles: [URL] = []

    private struct Intrinsics { let fx: Float; let fy: Float; let cx: Float; let cy: Float }

    private enum FuseError: LocalizedError {
        case notLoaded, noFace
        var errorDescription: String? {
            switch self {
            case .notLoaded: return "MediaPipe not loaded"
            case .noFace: return "NO FACE FOUND"
            }
        }
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else {
                DispatchQueue.main.async { self.status = "Camera permission denied" }
                return
            }
            self.queue.async {
                self.configure()
                self.session.startRunning()
                self.setupLandmarker()
            }
        }
    }

    private func setupLandmarker() {
        guard let path = Bundle.main.path(forResource: "face_landmarker", ofType: "task") else {
            DispatchQueue.main.async { self.saveMessage = "face_landmarker.task is not in the app bundle" }
            return
        }
        do {
            let options = FaceLandmarkerOptions()
            options.baseOptions.modelAssetPath = path
            options.runningMode = .image
            options.numFaces = 1
            landmarker = try FaceLandmarker(options: options)
        } catch {
            DispatchQueue.main.async { self.saveMessage = "MediaPipe failed to load: \(error.localizedDescription)" }
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

    // MARK: - Depth + MediaPipe fusion

    private func depthArray(_ map: CVPixelBuffer) -> (data: [Float], w: Int, h: Int) {
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        let base = CVPixelBufferGetBaseAddress(map)!
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
            for x in 0..<w { out[y * w + x] = row[x] }
        }
        return (out, w, h)
    }

    private func scaledIntrinsics(_ cal: AVCameraCalibrationData?, depthWidth: Int) -> Intrinsics? {
        guard let cal = cal else { return nil }
        let m = cal.intrinsicMatrix
        let ref = cal.intrinsicMatrixReferenceDimensions
        let s = Float(depthWidth) / Float(ref.width)
        return Intrinsics(fx: m.columns.0.x * s, fy: m.columns.1.y * s,
                          cx: m.columns.2.x * s, cy: m.columns.2.y * s)
    }

    // Median of valid depths in a 3x3 patch (meters), nil if none
    private func sampleDepth(_ d: [Float], _ w: Int, _ h: Int, _ u: Float, _ v: Float) -> Float? {
        guard u.isFinite, v.isFinite else { return nil }
        let ui = min(max(Int(u.rounded()), 0), w - 1)
        let vi = min(max(Int(v.rounded()), 0), h - 1)
        var vals: [Float] = []
        for y in max(vi - 1, 0)...min(vi + 1, h - 1) {
            for x in max(ui - 1, 0)...min(ui + 1, w - 1) {
                let z = d[y * w + x]
                if z.isFinite && z > 0 { vals.append(z) }
            }
        }
        if vals.isEmpty { return nil }
        vals.sort()
        let n = vals.count
        return n % 2 == 1 ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2
    }

    // Transposes the sensor-native frame so the face is upright for MediaPipe
    private func uprightBuffer(from pb: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let rotated = CIImage(cvPixelBuffer: pb).oriented(.leftMirrored)
        // keep the image's origin at (0, 0) so it renders into the new buffer
        let img = rotated.transformed(by: CGAffineTransform(translationX: -rotated.extent.origin.x,
                                                            y: -rotated.extent.origin.y))
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var created: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, h, w, kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary, &created)
        guard let out = created else { return nil }
        ciContext.render(img, to: out)
        return out
    }
    
    private func landmarks3D(pixelBuffer pb: CVPixelBuffer, depth: [Float], dw: Int, dh: Int,
                             k: Intrinsics) throws -> [SIMD3<Float>?] {
        guard let landmarker = landmarker else { throw FuseError.notLoaded }
        guard let upright = uprightBuffer(from: pb) else { throw FuseError.noFace }
        let image = try MPImage(pixelBuffer: upright)
        let result = try landmarker.detect(image: image)
        guard let lms = result.faceLandmarks.first else { throw FuseError.noFace }
        return lms.map { lm -> SIMD3<Float>? in
            // The upright image is a transpose, so its x and y axes are swapped
            // relative to the depth map: depth column = lm.y, depth row = lm.x
            let u = lm.y * Float(dw), v = lm.x * Float(dh)
            guard let z = sampleDepth(depth, dw, dh, u, v) else { return nil }
            return SIMD3<Float>((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z) * 1000
        }
    }
    
    // All depth pixels inside a box around the landmarks (mm), like clean_face.py
    private func faceCrop(depth: [Float], dw: Int, dh: Int, k: Intrinsics,
                          lm: [SIMD3<Float>]) -> [SIMD3<Float>] {
        guard !lm.isEmpty else { return [] }
        let minX = lm.map { $0.x }.min()! - 15, maxX = lm.map { $0.x }.max()! + 15
        let minY = lm.map { $0.y }.min()! - 15, maxY = lm.map { $0.y }.max()! + 15
        let minZ = lm.map { $0.z }.min()! - 40, maxZ = lm.map { $0.z }.max()! + 40
        var out: [SIMD3<Float>] = []
        for v in 0..<dh {
            for u in 0..<dw {
                let z = depth[v * dw + u]
                guard z.isFinite && z > 0 else { continue }
                let zm = z * 1000
                guard zm > minZ && zm < maxZ else { continue }
                let x = (Float(u) - k.cx) * zm / k.fx
                let y = (Float(v) - k.cy) * zm / k.fy
                if x > minX && x < maxX && y > minY && y < maxY {
                    out.append(SIMD3<Float>(x, y, zm))
                }
            }
        }
        return out
    }

    private func plyString(_ pts: [SIMD3<Float>]) -> String {
        var s = "ply\nformat ascii 1.0\nelement vertex \(pts.count)\n"
        s += "property float x\nproperty float y\nproperty float z\nend_header\n"
        for p in pts { s += String(format: "%.2f %.2f %.2f\n", p.x, p.y, p.z) }
        return s
    }

    // MARK: - Saving

    // Call only on `queue`.
    private func writeFrame(prefix: String, dir: URL) throws -> (files: [URL], report: String) {
        guard let depth = latestDepth, let pb = latestPixelBuffer else {
            throw NSError(domain: "save", code: 0, userInfo: [NSLocalizedDescriptionKey: "No frame yet"])
        }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // RGB as JPEG
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(ci, from: ci.extent),
              let jpg = UIImage(cgImage: cg).jpegData(compressionQuality: 0.92) else {
            throw NSError(domain: "save", code: 1, userInfo: [NSLocalizedDescriptionKey: "RGB encode failed"])
        }
        let rgbURL = dir.appendingPathComponent("\(prefix)rgb.jpg")
        try jpg.write(to: rgbURL)

        // Raw depth: float32 meters, row-major
        let (arr, w, h) = depthArray(depth.depthDataMap)
        let depthURL = dir.appendingPathComponent("\(prefix)depth_\(w)x\(h)_float32_m.bin")
        let depthData = arr.withUnsafeBufferPointer { Data(buffer: $0) }
        try depthData.write(to: depthURL)
        var files = [rgbURL, depthURL]

        // MediaPipe + fusion
        let k = scaledIntrinsics(depth.cameraCalibrationData, depthWidth: w)
        var pts: [SIMD3<Float>?] = []
        var report: String
        if let k = k {
            do {
                pts = try landmarks3D(pixelBuffer: pb, depth: arr, dw: w, dh: h, k: k)
                let valid = pts.compactMap { $0 }
                report = "\(valid.count)/\(pts.count) with depth"
                if pts.count > 263 {
                    if let nose = pts[1] { report += String(format: " | nose %.0f mm", nose.z) }
                    if let a = pts[33], let b = pts[263] {
                        report += String(format: " | eyes %.1f mm", simd_distance(a, b))
                    }
                }
            } catch {
                report = error.localizedDescription
            }
        } else {
            report = "no camera calibration data"
        }

        if !pts.isEmpty, let k = k {
            let valid = pts.compactMap { $0 }
            let jsonArr: [Any] = pts.map { (p: SIMD3<Float>?) -> Any in
                if let p = p { return [Double(p.x), Double(p.y), Double(p.z)] }
                return NSNull()
            }
            let lmURL = dir.appendingPathComponent("\(prefix)landmarks_3d_mm.json")
            try JSONSerialization.data(withJSONObject: jsonArr, options: []).write(to: lmURL)
            let lmPly = dir.appendingPathComponent("\(prefix)landmarks.ply")
            try plyString(valid).write(to: lmPly, atomically: true, encoding: .utf8)
            let crop = faceCrop(depth: arr, dw: w, dh: h, k: k, lm: valid)
            let cropPly = dir.appendingPathComponent("\(prefix)face_crop.ply")
            try plyString(crop).write(to: cropPly, atomically: true, encoding: .utf8)
            files += [lmURL, lmPly, cropPly]
        }

        // Intrinsics + metadata
        var meta: [String: Any] = [
            "pose": prefix.isEmpty ? "single" : String(prefix.dropLast()),
            "timestamp": Date().timeIntervalSince1970,
            "depth_width": w, "depth_height": h,
            "rgb_width": CVPixelBufferGetWidth(pb), "rgb_height": CVPixelBufferGetHeight(pb),
            "depth_filtered": depth.isDepthDataFiltered,
            "mediapipe_report": report
        ]
        if let cal = depth.cameraCalibrationData {
            let m = cal.intrinsicMatrix
            let d = cal.intrinsicMatrixReferenceDimensions
            meta["fx"] = Double(m.columns.0.x); meta["fy"] = Double(m.columns.1.y)
            meta["cx"] = Double(m.columns.2.x); meta["cy"] = Double(m.columns.2.y)
            meta["intrinsics_ref_width"] = Double(d.width)
            meta["intrinsics_ref_height"] = Double(d.height)
        }
        let metaURL = dir.appendingPathComponent("\(prefix)meta.json")
        try JSONSerialization.data(withJSONObject: meta, options: .prettyPrinted).write(to: metaURL)
        files.append(metaURL)

        return (files, report)
    }

    private func newSessionDir(_ name: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("\(name)_\(Int(Date().timeIntervalSince1970))")
    }

    // Single frame
    func saveFrame() {
        let dir = newSessionDir("capture")
        queue.async {
            do {
                let result = try self.writeFrame(prefix: "", dir: dir)
                DispatchQueue.main.async {
                    self.saveMessage = "\(result.report)\nSaved \(dir.lastPathComponent)"
                    self.savedFiles = result.files
                }
            } catch {
                DispatchQueue.main.async { self.saveMessage = "Save failed: \(error.localizedDescription)" }
            }
        }
    }

    // Guided capture: front, left, right, each after a 4-second countdown
    func startGuidedCapture() {
        guard !isCapturing else { return }
        isCapturing = true
        savedFiles = []
        saveMessage = ""
        pending = 3
        let dir = newSessionDir("face")
        let poses: [(String, String)] = [
            ("front", "Look straight at the phone"),
            ("left", "Slowly turn your head to YOUR LEFT (about 30°)"),
            ("right", "Slowly turn your head to YOUR RIGHT (about 30°)")
        ]
        var t = 0.5
        for (name, text) in poses {
            for s in stride(from: 4, through: 1, by: -1) {
                DispatchQueue.main.asyncAfter(deadline: .now() + t) { self.prompt = "\(text)\n\(s)" }
                t += 1.0
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + t) { self.prompt = "Hold still..." }
            t += 0.4
            DispatchQueue.main.asyncAfter(deadline: .now() + t) {
                self.prompt = "Processing \(name)..."
                self.capturePose(name, in: dir)
            }
            t += 0.8
        }
    }

    private func capturePose(_ name: String, in dir: URL) {
        queue.async {
            let outcome: (files: [URL], line: String)
            do {
                let r = try self.writeFrame(prefix: "\(name)_", dir: dir)
                outcome = (r.files, "\(name): \(r.report)")
            } catch {
                outcome = ([], "\(name) failed: \(error.localizedDescription)")
            }
            DispatchQueue.main.async {
                self.savedFiles.append(contentsOf: outcome.files)
                self.saveMessage += outcome.line + "\n"
                self.pending -= 1
                if self.pending == 0 {
                    self.prompt = ""
                    self.isCapturing = false
                    self.saveMessage += "Saved \(dir.lastPathComponent)"
                }
            }
        }
    }
}
