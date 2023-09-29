var video = document.querySelector('img');

function updateStream() {
    video.src = "{{ url_for('video_feed') }}";
    setTimeout(updateStream, 1000); // Refresh the stream every 1 second
}

updateStream();