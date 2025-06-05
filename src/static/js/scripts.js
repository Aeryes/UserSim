document.addEventListener("DOMContentLoaded", function () {
    const modelSelect = document.getElementById('modelSelect');
    const startButton = document.getElementById('startTrainingBtn');

    if (modelSelect && startButton) {
        modelSelect.addEventListener('change', () => {
            startButton.disabled = !modelSelect.value;
        });
    }

    const modelIdInput = document.querySelector("input[name='hf_model_id']");
    const downloadForm = document.querySelector("form[action='/v1/training-dashboard/download-model']");
    const downloadMessage = document.getElementById("download-message");

    const logContainer = document.getElementById("log-container");
    if (logContainer) {
        const eventSource = new EventSource("/v1/training-dashboard/stream");
        eventSource.onmessage = function (event) {
            logContainer.textContent += event.data + "\n";
            logContainer.scrollTop = logContainer.scrollHeight;
        };
    }

    function checkStatus(modelId) {
        fetch(`/v1/training-dashboard/download-status?model_id=${encodeURIComponent(modelId)}`)
            .then(res => res.json())
            .then(data => {
                if (data && data[modelId]) {
                    const { status, error } = data[modelId];

                    if (status === "complete") {
                        downloadMessage.textContent = "✅ Download Complete";
                        downloadMessage.style.color = "green";
                        setTimeout(() => {
                            downloadMessage.style.display = "none";
                        }, 5000);
                    } else if (status === "error") {
                        downloadMessage.textContent = `❌ Error: ${error}`;
                        downloadMessage.style.color = "red";
                    } else if (status === "downloading") {
                        setTimeout(() => checkStatus(modelId), 2000);
                    }
                }
            })
            .catch(err => {
                downloadMessage.textContent = `❌ Failed to check status: ${err.message}`;
                downloadMessage.style.color = "red";
            });
    }

    if (downloadForm && modelIdInput && downloadMessage) {
        downloadForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const modelId = modelIdInput.value.trim();
            const displayName = downloadForm.querySelector("input[name='display_name']").value.trim();

            if (modelId.length > 0 && displayName.length > 0) {
                downloadMessage.textContent = "⏳ Downloading... Please do not refresh or leave the page.";
                downloadMessage.style.color = "green";
                downloadMessage.style.display = "block";

                fetch('/v1/training-dashboard/download-model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ hf_model_id: modelId, display_name: displayName })
                }).then(() => {
                    setTimeout(() => checkStatus(modelId), 2000);
                }).catch(err => {
                    downloadMessage.textContent = `❌ Error: ${err.message}`;
                    downloadMessage.style.color = 'red';
                });
            }
        });
    }
});
