const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const preview = document.getElementById("preview");
const predictButton = document.getElementById("predictButton");

const loading = document.getElementById("loading");
const result = document.getElementById("result");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");


// Image selection
imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    fileName.textContent = file.name;

    const imageURL = URL.createObjectURL(file);

    preview.src = imageURL;
    preview.style.display = "block";

    result.style.display = "none";
});


// Prediction
predictButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an animal image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    predictButton.disabled = true;
    loading.style.display = "block";
    result.style.display = "none";

    try {

        console.log("Sending image to API...");

        const response = await fetch(
           "http://127.0.0.1:8000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        console.log("API status:", response.status);

        const data = await response.json();

        console.log("API response:", data);

        if (!response.ok) {
            throw new Error(
                data.detail || "Prediction failed."
            );
        }

        prediction.textContent = data.prediction;
        confidence.textContent = data.confidence + "%";

        result.style.display = "block";

    } catch (error) {

        console.error("ERROR:", error);

        alert(
            "Error: " + error.message
        );

    } finally {

        predictButton.disabled = false;
        loading.style.display = "none";
    }
});