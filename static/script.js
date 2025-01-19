let videoElement = document.getElementById('video');
let streamStarted = false;

function startStream() {
    if (!streamStarted) {
        videoElement.src = "/video_feed";  // Start the stream
        videoElement.style.display = "block";  // Make sure the video is visible
        streamStarted = true;
    }
}

function stopStream() {
    // Stop the video stream
    videoElement.src = "";  // Stop the video stream by clearing the source
    videoElement.style.display = "none";  // Hide the video element
    streamStarted = false;
}
