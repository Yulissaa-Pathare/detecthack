const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const resultDiv = document.getElementById("result");

uploadForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload a video or image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    resultDiv.innerHTML = "Analyzing...";

    try {
        const response = await fetch("https://detecthack-api.onrender.com/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.result) {
            resultDiv.innerHTML = `Result: ${data.result}`;
        } else {
            resultDiv.innerHTML = "No result returned.";
        }

    } catch (error) {
        console.error(error);
        resultDiv.innerHTML = "Error analyzing file.";
    }
});