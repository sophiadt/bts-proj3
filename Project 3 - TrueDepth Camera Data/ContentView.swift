import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let img = camera.depthImage {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            VStack {
                if !camera.prompt.isEmpty {
                    Text(camera.prompt)
                        .font(.title2.bold())
                        .multilineTextAlignment(.center)
                        .padding()
                        .background(.black.opacity(0.7))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .padding(.top, 60)
                }
                Spacer()
                VStack(spacing: 8) {
                    Text(camera.status)
                    Text(camera.saveMessage)
                    HStack {
                        Button("Guided capture") { camera.startGuidedCapture() }
                            .buttonStyle(.borderedProminent)
                            .disabled(camera.isCapturing)
                        Button("Save frame") { camera.saveFrame() }
                            .buttonStyle(.bordered)
                            .disabled(camera.isCapturing)
                        if !camera.savedFiles.isEmpty && !camera.isCapturing {
                            ShareLink(items: camera.savedFiles) {
                                Label("Share", systemImage: "square.and.arrow.up")
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                .font(.system(.footnote, design: .monospaced))
                .padding(8)
                .background(.black.opacity(0.6))
                .foregroundStyle(.white)
                .padding()
            }
        }
        .onAppear { camera.start() }
    }
}
