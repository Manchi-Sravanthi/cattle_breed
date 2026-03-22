const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const imagePreview = document.getElementById("imagePreview");
const resultDiv = document.getElementById("result");

let selectedFile = null;

// File select
fileInput.addEventListener("change", (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        const reader = new FileReader();
        reader.onload = function(event) {
            imagePreview.innerHTML = `<img src="${event.target.result}" />`;
            resultDiv.innerHTML = "";
        };
        reader.readAsDataURL(selectedFile);
    }
});

// Predict button click
predictBtn.addEventListener("click", () => {
    if (!selectedFile) {
        alert("Please select an image!");
        return;
    }
    const formData = new FormData();
    formData.append("file", selectedFile);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if(data.error){
            resultDiv.innerHTML = `<p>${data.error}</p>`;
        } else {
            imagePreview.innerHTML = `<img src="${data.image}" />`;
            resultDiv.innerHTML = `<p>✅ Prediction: ${data.prediction}</p>
                                   <p>📊 Confidence: ${data.confidence}</p>`;
        }
    })
    .catch(err => console.error(err));
});