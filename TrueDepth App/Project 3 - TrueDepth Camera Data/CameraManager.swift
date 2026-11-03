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
    private var videoLandmarker: FaceLandmarker?
    private var recording = false            // only touched on visionQueue
    private let visionQueue = DispatchQueue(label: "vision.queue", qos: .userInitiated)
    private var visionBusy = false            // only touched on `queue`
    private var wantRGB = false
    private var recStart: Double = 0
    private var lastTimestampMs = -1
    private var recMisses = 0
    private var recFrames: [(t: Double, pts: [SIMD3<Float>?], bs: [Float], raw: Int?)] = []
    private var recIntrinsics: Intrinsics?
    private let ciContext = CIContext()
    // raw RGB + depth saving while recording (all of these are only touched on visionQueue)
    private var recDir: URL?
    private var recRaw = false
    private var recBlendNames: [String] = []
    private var recMeta: [String: Any] = [:]
    private var rawSeen = 0
    private var rawSaved = 0
    private var rawDropped = 0
    private var rawPending = 0
    private let rawEvery = 2                 // save every 2nd processed frame
    private let rawQueue = DispatchQueue(label: "raw.queue", qos: .utility)
    private let rawCI = CIContext()
    private var wantRaw = false

    @Published var status = "Starting..."
    @Published var saveMessage = ""
    @Published var prompt = ""
    @Published var isCapturing = false
    @Published var isRecording = false
    @Published var meshPoints: [CGPoint] = []   // live face landmarks, 0...1 in the upright image
    @Published var rgbImage: UIImage?
    @Published var meshValid: [Bool] = []       // false = no depth (or outside the frame) for that landmark
    @Published var faceStatus = "NO FACE"
    @Published var faceOK = false
    @Published var showRGB = false { didSet { wantRGB = showRGB } }
    @Published var saveRaw = false { didSet { wantRaw = saveRaw } }
    lazy var meshEdges: [(Int, Int)] = FaceLandmarker.tesselationConnections().map { (Int($0.start), Int($0.end)) }
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
            options.outputFaceBlendshapes = true
            landmarker = try FaceLandmarker(options: options)
            let vopts = FaceLandmarkerOptions()
            vopts.baseOptions.modelAssetPath = path
            vopts.runningMode = .video
            vopts.numFaces = 1
            vopts.outputFaceBlendshapes = true
            videoLandmarker = try FaceLandmarker(options: vopts)
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
        // Face detection runs on its own queue so it never blocks the camera.
        // If it is still busy with the previous frame, this frame is skipped.
        if !visionBusy, let pb = latestPixelBuffer {
            visionBusy = true
            let t = CMTimeGetSeconds(syncedDepth.timestamp)
            let frameDepth = depth
            visionQueue.async {
                self.processVision(depth: frameDepth, pixelBuffer: pb, time: t)
                self.queue.async { self.visionBusy = false }
            }
        }

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
        let image = (frameCount % 3 == 0 && !wantRGB) ? makeDepthImage(map) : nil
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

    // Median of valid depths in a 3x3 patch (meters). If that patch has no depth
    // (eyes, nostrils, hair edge), look in a 7x7 patch. nil if there is none.
    private func sampleDepth(_ d: [Float], _ w: Int, _ h: Int, _ u: Float, _ v: Float) -> Float? {
        guard u.isFinite, v.isFinite else { return nil }
        let ui = min(max(Int(u.rounded()), 0), w - 1)
        let vi = min(max(Int(v.rounded()), 0), h - 1)
        for r in [1, 3] {
            var vals: [Float] = []
            for y in max(vi - r, 0)...min(vi + r, h - 1) {
                for x in max(ui - r, 0)...min(ui + r, w - 1) {
                    let z = d[y * w + x]
                    if z.isFinite && z > 0 { vals.append(z) }
                }
            }
            if !vals.isEmpty {
                vals.sort()
                let n = vals.count
                return n % 2 == 1 ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2
            }
        }
        return nil
    }

    // The 52 blendshape scores MediaPipe returns with the landmarks (names + scores, same order)
    private func blendshapes(_ result: FaceLandmarkerResult) -> (names: [String], scores: [Float])? {
        guard let cats = result.faceBlendshapes.first?.categories, !cats.isEmpty else { return nil }
        return (cats.map { $0.categoryName ?? "" }, cats.map { $0.score })
    }

    // Distance between the pupils in mm. Used to check the depth scale (adults are roughly 54-74 mm).
    // Falls back to the midpoints of the eye corners if the iris points have no depth.
    private func pupilDistance(_ pts: [SIMD3<Float>?]) -> (mm: Float, source: String)? {
        guard pts.count > 473 else { return nil }
        if let a = pts[468], let b = pts[473] { return (simd_distance(a, b), "iris") }
        if let a1 = pts[33], let a2 = pts[133], let b1 = pts[362], let b2 = pts[263] {
            return (simd_distance((a1 + a2) / 2, (b1 + b2) / 2), "eye corners (approx)")
        }
        return nil
    }

    // Lens distortion data from the camera calibration (saved only, never applied)
    private func lensMeta(_ cal: AVCameraCalibrationData?) -> [String: Any] {
        guard let cal = cal else { return [:] }
        func floats(_ d: Data?) -> [Double] {
            guard let d = d else { return [] }
            return d.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: Float32.self)).map { Double($0) }
            }
        }
        return [
            "lens_distortion_center_x": Double(cal.lensDistortionCenter.x),
            "lens_distortion_center_y": Double(cal.lensDistortionCenter.y),
            "lens_distortion_lookup": floats(cal.lensDistortionLookupTable),
            "lens_distortion_inverse_lookup": floats(cal.inverseLensDistortionLookupTable),
            "pixel_size_mm": Double(cal.pixelSize)
        ]
    }

    // Transposes the sensor-native frame so the face is upright for MediaPipe.
    // maxSide shrinks the result (landmarks are normalized, so size does not matter).
    private func uprightBuffer(from pb: CVPixelBuffer, maxSide: CGFloat? = nil) -> CVPixelBuffer? {
        var img = CIImage(cvPixelBuffer: pb).oriented(.leftMirrored)
        img = img.transformed(by: CGAffineTransform(translationX: -img.extent.origin.x,
                                                    y: -img.extent.origin.y))
        if let m = maxSide {
            let sc = m / max(img.extent.width, img.extent.height)
            if sc < 1 { img = img.transformed(by: CGAffineTransform(scaleX: sc, y: sc)) }
        }
        let ow = Int(img.extent.width.rounded()), oh = Int(img.extent.height.rounded())
        var created: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, ow, oh, kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary, &created)
        guard let out = created else { return nil }
        ciContext.render(img, to: out)
        return out
    }

    private func landmarks3D(pixelBuffer pb: CVPixelBuffer, depth: [Float], dw: Int, dh: Int,
                             k: Intrinsics) throws -> (pts: [SIMD3<Float>?], blend: (names: [String], scores: [Float])?) {
        guard let landmarker = landmarker else { throw FuseError.notLoaded }
        guard let upright = uprightBuffer(from: pb) else { throw FuseError.noFace }
        let image = try MPImage(pixelBuffer: upright)
        let result = try landmarker.detect(image: image)
        guard let lms = result.faceLandmarks.first else { throw FuseError.noFace }
        let points = lms.map { lm -> SIMD3<Float>? in
            // The upright image is a transpose, so its x and y axes are swapped
            // relative to the depth map: depth column = lm.y, depth row = lm.x
            let u = lm.y * Float(dw), v = lm.x * Float(dh)
            guard let z = sampleDepth(depth, dw, dh, u, v) else { return nil }
            return SIMD3<Float>((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z) * 1000
        }
        return (points, blendshapes(result))
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
        var blend: (names: [String], scores: [Float])?
        var report: String
        if let k = k {
            do {
                let fused = try landmarks3D(pixelBuffer: pb, depth: arr, dw: w, dh: h, k: k)
                pts = fused.pts
                blend = fused.blend
                let valid = pts.compactMap { $0 }
                report = "\(valid.count)/\(pts.count) with depth"
                if pts.count > 263 {
                    if let nose = pts[1] { report += String(format: " | nose %.0f mm", nose.z) }
                    if let a = pts[33], let b = pts[263] {
                        report += String(format: " | eyes %.1f mm", simd_distance(a, b))
                    }
                    if let p = pupilDistance(pts) {
                        report += String(format: " | pupils %.1f mm", p.mm)
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
            meta.merge(lensMeta(cal)) { a, _ in a }
        }
        if let p = pupilDistance(pts) {
            meta["pupil_distance_mm"] = Double(p.mm)
            meta["pupil_source"] = p.source
        }
        if let b = blend {
            var dict: [String: Double] = [:]
            for (n, sc) in zip(b.names, b.scores) { dict[n] = Double(sc) }
            meta["blendshapes"] = dict
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

    // MARK: - Live vision: face mesh, status label, recording. Runs on `visionQueue`.

    private func processVision(depth: AVDepthData, pixelBuffer pb: CVPixelBuffer, time: Double) {
        let ms = Int(time * 1000)
        guard ms > lastTimestampMs, let vl = videoLandmarker,
              let small = uprightBuffer(from: pb, maxSide: 640) else { return }
        lastTimestampMs = ms

        var lms: [NormalizedLandmark] = []
        var bs: (names: [String], scores: [Float])?
        if let image = try? MPImage(pixelBuffer: small),
           let result = try? vl.detect(videoFrame: image, timestampInMilliseconds: ms),
           let first = result.faceLandmarks.first {
            lms = first
            bs = blendshapes(result)
        }
        let norm = lms.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }

        // 3D positions in mm, and how many landmarks have a valid depth reading
        var pts: [SIMD3<Float>?] = []
        var withDepth = 0
        var inFrame = true
        var valid: [Bool] = []
        var kUsed: Intrinsics?
        var depthArr: [Float] = []
        var dW = 0, dH = 0
        if !lms.isEmpty {
            let (arr, w, h) = depthArray(depth.depthDataMap)
            depthArr = arr; dW = w; dH = h
            kUsed = scaledIntrinsics(depth.cameraCalibrationData, depthWidth: w)
            for lm in lms {
                let inside = lm.x >= 0 && lm.x <= 1 && lm.y >= 0 && lm.y <= 1
                if !inside { inFrame = false }
                let u = lm.y * Float(w), v = lm.x * Float(h)
                if inside, let k = kUsed, let z = sampleDepth(arr, w, h, u, v) {
                    pts.append(SIMD3<Float>((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z) * 1000)
                    withDepth += 1
                    valid.append(true)
                } else {
                    pts.append(nil)
                    valid.append(false)
                }
            }
        }

        var recCount = 0
        if recording {
            if lms.isEmpty {
                recMisses += 1
            } else {
                if recFrames.isEmpty {
                    recStart = time
                    recIntrinsics = kUsed
                    var m: [String: Any] = [
                        "depth_width": dW, "depth_height": dH,
                        "rgb_width": CVPixelBufferGetWidth(pb), "rgb_height": CVPixelBufferGetHeight(pb),
                        "depth_filtered": depth.isDepthDataFiltered
                    ]
                    if let cal = depth.cameraCalibrationData {
                        let im = cal.intrinsicMatrix
                        let d = cal.intrinsicMatrixReferenceDimensions
                        m["fx"] = Double(im.columns.0.x); m["fy"] = Double(im.columns.1.y)
                        m["cx"] = Double(im.columns.2.x); m["cy"] = Double(im.columns.2.y)
                        m["intrinsics_ref_width"] = Double(d.width)
                        m["intrinsics_ref_height"] = Double(d.height)
                        m.merge(lensMeta(cal)) { a, _ in a }
                    }
                    recMeta = m
                }
                if recBlendNames.isEmpty, let b = bs { recBlendNames = b.names }
                var rawIndex: Int?
                if recRaw, let dir = recDir {
                    rawSeen += 1
                    if rawSeen % rawEvery == 0 {
                        rawIndex = queueRawFrame(dir: dir.appendingPathComponent("raw"),
                                                 depth: depthArr, w: dW, h: dH, pb: pb)
                    }
                }
                recFrames.append((time - recStart, pts, bs?.scores ?? [], rawIndex))
            }
            recCount = recFrames.count
        }

        let label: String
        let ok: Bool
        if lms.isEmpty {
            label = "NO FACE"; ok = false
        } else if !inFrame {
            label = "\(withDepth)/\(lms.count) · FACE CUT OFF"; ok = false
        } else {
            label = "\(withDepth)/\(lms.count)"; ok = (withDepth == lms.count)
        }

        var photo: UIImage?
        if wantRGB {
            let ci = CIImage(cvPixelBuffer: small)
            if let cg = ciContext.createCGImage(ci, from: ci.extent) { photo = UIImage(cgImage: cg) }
        }
        let misses = recMisses
        let isRec = recording
        DispatchQueue.main.async {
            self.meshPoints = norm
            self.meshValid = valid
            self.faceStatus = label
            self.faceOK = ok
            if let photo = photo { self.rgbImage = photo }
            if isRec && recCount % 10 == 0 {
                self.saveMessage = "Recording: \(recCount) frames, \(misses) without a face"
            }
        }
    }

    // Saves one raw RGB + depth frame in the background. Returns its index, or nil if skipped (writer busy).
    // Call only on `visionQueue`.
    private func queueRawFrame(dir: URL, depth: [Float], w: Int, h: Int, pb: CVPixelBuffer) -> Int? {
        if rawPending >= 3 { rawDropped += 1; return nil }
        rawPending += 1
        let index = rawSaved
        rawSaved += 1
        rawQueue.async {
            let name = String(format: "frame_%05d", index)
            let ci = CIImage(cvPixelBuffer: pb)
            if let cg = self.rawCI.createCGImage(ci, from: ci.extent),
               let jpg = UIImage(cgImage: cg).jpegData(compressionQuality: 0.9) {
                try? jpg.write(to: dir.appendingPathComponent("\(name)_rgb.jpg"))
            }
            let data = depth.withUnsafeBufferPointer { Data(buffer: $0) }
            try? data.write(to: dir.appendingPathComponent("\(name)_depth_\(w)x\(h)_float32_m.bin"))
            self.visionQueue.async { self.rawPending -= 1 }
        }
        return index
    }

    // Zips a folder (so hundreds of raw frames can be shared as one file)
    private func zipFolder(_ dir: URL) -> URL? {
        var result: URL?
        var error: NSError?
        NSFileCoordinator().coordinate(readingItemAt: dir, options: [.forUploading], error: &error) { zipped in
            let dest = dir.deletingLastPathComponent().appendingPathComponent(dir.lastPathComponent + ".zip")
            try? FileManager.default.removeItem(at: dest)
            if (try? FileManager.default.copyItem(at: zipped, to: dest)) != nil { result = dest }
        }
        return result
    }

    func startRecording() {
        guard !isRecording, !isCapturing else { return }
        isRecording = true
        saveMessage = "Recording... start with a neutral face"
        savedFiles = []
        visionQueue.async {
            self.recFrames = []
            self.recMisses = 0
            self.recBlendNames = []
            self.recMeta = [:]
            self.rawSeen = 0; self.rawSaved = 0; self.rawDropped = 0; self.rawPending = 0
            self.recRaw = self.wantRaw
            let dir = self.newSessionDir("recording")
            self.recDir = dir
            let rawDir = dir.appendingPathComponent("raw")
            try? FileManager.default.createDirectory(at: self.recRaw ? rawDir : dir, withIntermediateDirectories: true)
            self.recording = true
        }
    }

    func stopRecording() {
        guard isRecording else { return }
        visionQueue.async {
            self.recording = false
            self.rawQueue.sync { }                 // wait until every queued raw frame is written
            let frames = self.recFrames, misses = self.recMisses, k = self.recIntrinsics
            let dir = self.recDir ?? self.newSessionDir("recording")
            let usedRaw = self.recRaw
            let savedRaw = self.rawSaved, droppedRaw = self.rawDropped
            var message: String
            var files: [URL] = []
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                func r2(_ v: Float) -> Double { (Double(v) * 100).rounded() / 100 }
                let frameJSON: [Any] = frames.map { f in
                    let pts: [Any] = f.pts.map { p -> Any in
                        if let p = p { return [r2(p.x), r2(p.y), r2(p.z)] }
                        return NSNull()
                    }
                    var entry: [String: Any] = ["t": (f.t * 1000).rounded() / 1000, "pts": pts]
                    if !f.bs.isEmpty { entry["bs"] = f.bs.map { (Double($0) * 1000).rounded() / 1000 } }
                    if let r = f.raw { entry["raw"] = r }
                    return entry
                }
                var root: [String: Any] = ["units": "mm", "frame_count": frames.count,
                                           "frames_without_face": misses, "frames": frameJSON]
                if !self.recBlendNames.isEmpty { root["blendshape_names"] = self.recBlendNames }
                if let k = k {
                    root["fx"] = Double(k.fx); root["fy"] = Double(k.fy)
                    root["cx"] = Double(k.cx); root["cy"] = Double(k.cy)
                }
                let url = dir.appendingPathComponent("landmarks_sequence_mm.json")
                try JSONSerialization.data(withJSONObject: root, options: []).write(to: url)
                files = [url]

                // camera numbers, lens distortion and pupil distance (depth-scale check)
                var meta = self.recMeta
                meta["raw_frames_saved"] = savedRaw
                meta["raw_frames_dropped"] = droppedRaw
                meta["raw_every_nth_frame"] = self.rawEvery
                let ipds = frames.compactMap { self.pupilDistance($0.pts) }
                let irisIPDs = ipds.filter { $0.source == "iris" }.map { $0.mm }.sorted()
                let anyIPDs = ipds.map { $0.mm }.sorted()
                var pupilNote = ""
                if !irisIPDs.isEmpty {
                    meta["pupil_distance_mm_median"] = Double(irisIPDs[irisIPDs.count / 2])
                    meta["pupil_source"] = "iris"
                    pupilNote = String(format: " | pupils %.1f mm", irisIPDs[irisIPDs.count / 2])
                } else if !anyIPDs.isEmpty {
                    meta["pupil_distance_mm_median"] = Double(anyIPDs[anyIPDs.count / 2])
                    meta["pupil_source"] = "eye corners (approx)"
                    pupilNote = String(format: " | pupils ~%.1f mm", anyIPDs[anyIPDs.count / 2])
                }
                let metaURL = dir.appendingPathComponent("recording_meta.json")
                try JSONSerialization.data(withJSONObject: meta, options: .prettyPrinted).write(to: metaURL)
                files.append(metaURL)

                var rawNote = ""
                if usedRaw {
                    if let zip = self.zipFolder(dir) { files.insert(zip, at: 0) }
                    rawNote = "\nRaw frames: \(savedRaw) saved, \(droppedRaw) skipped (writer busy)"
                }
                let secs = frames.last?.t ?? 0
                message = String(format: "Saved %d frames over %.1f s (%.0f fps), %d without a face%@%@\n%@",
                                 frames.count, secs, secs > 0 ? Double(frames.count) / secs : 0,
                                 misses, pupilNote, rawNote, dir.lastPathComponent)
            } catch {
                message = "Save failed: \(error.localizedDescription)"
            }
            DispatchQueue.main.async {
                self.isRecording = false
                self.saveMessage = message
                self.savedFiles = files
            }
        }
    }
}
