import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()

    var body: some View {
        ZStack(alignment: .bottom) {
            Color.black.ignoresSafeArea()
            if let img = camera.depthImage {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            VStack(spacing: 8) {
                Text(camera.status)
                Text(camera.saveMessage)
                HStack {
                    Button("Save frame") { camera.saveFrame() }
                        .buttonStyle(.borderedProminent)
                    if !camera.savedFiles.isEmpty {
                        ShareLink(items: camera.savedFiles) {
                            Label("Share files", systemImage: "square.and.arrow.up")
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
        .onAppear { camera.start() }
    }
}
