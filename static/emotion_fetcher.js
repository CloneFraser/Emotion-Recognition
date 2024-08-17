function fetchEmotion() {
        fetch('/current_emotion')
            .then(response => response.json())
            .then(data => {
                document.getElementById('emotion').innerText = 'Emotion: ' + data.emotion;
            })
            .catch(console.error);
}

setInterval(fetchEmotion, 100);