import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var recordMode = false          // false = photo, true = record
    @State private var flash = false
    @State private var recordStartedAt: Date?

    var body: some View {
        VStack(spacing: 0) {
            preview
            Spacer(minLength: 0)
            infoText
            modeSelector
                .padding(.top, 14)
            Spacer(minLength: 0)
            shutterRow
                .padding(.bottom, 24)
        }
        .background(Color.black.ignoresSafeArea())
        .onAppear { camera.start() }
    }

    // MARK: - Viewfinder (3:4, full width, like the Camera app)

    private var preview: some View {
        ZStack {
            Color.black
            if let img = (camera.showRGB ? camera.rgbImage : camera.depthImage) {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFit()
                    .overlay(MeshOverlay(points: camera.meshPoints, valid: camera.meshValid, edges: camera.meshEdges))
            }
        }
        .frame(maxWidth: .infinity)
        .aspectRatio(3.0 / 4.0, contentMode: .fit)
        .clipped()
        .overlay(alignment: .topLeading) {
            Button { camera.showRGB.toggle() } label: {
                HStack(spacing: 5) {
                    Image(systemName: camera.showRGB ? "camera" : "dot.viewfinder")
                    Text(camera.showRGB ? "CAMERA" : "DEPTH")
                }
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.ultraThinMaterial, in: Capsule())
            }
            .padding(12)
        }
        .overlay(alignment: .topTrailing) {
            pill(camera.faceStatus,
                 color: camera.faceOK ? .green : (camera.faceStatus == "NO FACE" ? .red : .orange))
                .padding(12)
        }
        .overlay(alignment: .top) {
            if camera.isRecording, let start = recordStartedAt {
                TimelineView(.periodic(from: start, by: 1)) { ctx in
                    let s = max(0, Int(ctx.date.timeIntervalSince(start)))
                    Text(String(format: "%02d:%02d", s / 60, s % 60))
                        .font(.system(size: 15, weight: .semibold, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 5)
                        .background(Color.red, in: Capsule())
                }
                .padding(.top, 14)
            }
        }
        .overlay(alignment: .bottom) {
            if !camera.prompt.isEmpty {
                Text(camera.prompt)
                    .font(.title3.bold())
                    .multilineTextAlignment(.center)
                    .foregroundColor(.white)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 12)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                    .padding(.bottom, 16)
            }
        }
        .overlay(
            Color.white.opacity(flash ? 0.7 : 0)
                .allowsHitTesting(false)
                .animation(.easeOut(duration: 0.12), value: flash)
        )
    }

    private func pill(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 14, weight: .semibold, design: .monospaced))
            .foregroundColor(color)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
    }

    // MARK: - Status text and mode labels

    private var infoText: some View {
        VStack(spacing: 3) {
            Text(camera.saveMessage.isEmpty ? camera.status : camera.saveMessage)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(Color.white.opacity(0.6))
                .multilineTextAlignment(.center)
                .lineLimit(3)
                .minimumScaleFactor(0.8)
        }
        .padding(.horizontal, 20)
        .padding(.top, 10)
        .frame(minHeight: 44)
    }

    private var modeSelector: some View {
        HStack(spacing: 28) {
            modeLabel("PHOTO", selected: !recordMode) { recordMode = false }
            modeLabel("RECORD", selected: recordMode) { recordMode = true }
        }
    }

    private func modeLabel(_ title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(selected ? Color.yellow : Color.white)
        }
        .buttonStyle(.plain)
        .disabled(camera.isRecording || camera.isCapturing)
    }

    // MARK: - Bottom row: guided capture, shutter, share

    private var shutterRow: some View {
        HStack {
            Button { camera.startGuidedCapture() } label: {
                Image(systemName: "viewfinder")
                    .font(.system(size: 22))
                    .foregroundColor(.white)
                    .frame(width: 54, height: 54)
                    .background(Color.white.opacity(0.18), in: Circle())
            }
            .disabled(camera.isCapturing || camera.isRecording)

            Spacer()

            ShutterButton(recordMode: recordMode, isRecording: camera.isRecording) {
                if recordMode {
                    if camera.isRecording {
                        camera.stopRecording()
                        recordStartedAt = nil
                    } else {
                        camera.startRecording()
                        recordStartedAt = Date()
                    }
                } else {
                    flash = true
                    camera.saveFrame()
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) { flash = false }
                }
            }
            .disabled(camera.isCapturing)

            Spacer()

            Group {
                if camera.savedFiles.isEmpty {
                    Color.clear
                } else {
                    ShareLink(items: camera.savedFiles) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 22))
                            .foregroundColor(.white)
                            .frame(width: 54, height: 54)
                            .background(Color.white.opacity(0.18), in: Circle())
                    }
                }
            }
            .frame(width: 54, height: 54)
        }
        .padding(.horizontal, 30)
    }
}

// Round shutter like the Camera app: white ring with a white disc (photo),
// red disc (record), red rounded square while recording.
struct ShutterButton: View {
    let recordMode: Bool
    let isRecording: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().strokeBorder(Color.white, lineWidth: 4).frame(width: 76, height: 76)
                if recordMode {
                    RoundedRectangle(cornerRadius: isRecording ? 7 : 31)
                        .fill(Color.red)
                        .frame(width: isRecording ? 30 : 62, height: isRecording ? 30 : 62)
                } else {
                    Circle().fill(Color.white).frame(width: 62, height: 62)
                }
            }
        }
        .buttonStyle(.plain)
        .animation(.easeInOut(duration: 0.15), value: isRecording)
        .animation(.easeInOut(duration: 0.15), value: recordMode)
    }
}

// Wireframe of the detected face. Blue where depth is available, red where it is not.
struct MeshOverlay: View {
    let points: [CGPoint]
    let valid: [Bool]
    let edges: [(Int, Int)]

    var body: some View {
        Canvas { context, size in
            guard points.count > 468, valid.count == points.count else { return }
            func pt(_ i: Int) -> CGPoint {
                CGPoint(x: points[i].x * size.width, y: points[i].y * size.height)
            }
            var good = Path()
            var bad = Path()
            for (a, b) in edges where a < points.count && b < points.count {
                if valid[a] && valid[b] {
                    good.move(to: pt(a)); good.addLine(to: pt(b))
                } else {
                    bad.move(to: pt(a)); bad.addLine(to: pt(b))
                }
            }
            context.stroke(good, with: .color(Color(red: 0.35, green: 0.45, blue: 1.0)), lineWidth: 0.8)
            context.stroke(bad, with: .color(.red), lineWidth: 1.2)
            for i in 0..<points.count where !valid[i] {
                let p = pt(i)
                context.fill(Path(ellipseIn: CGRect(x: p.x - 2.5, y: p.y - 2.5, width: 5, height: 5)),
                             with: .color(.red))
            }
        }
        .allowsHitTesting(false)
    }
}
