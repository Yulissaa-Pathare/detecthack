async function analyzeVideo() {

    const fileInput = document.getElementById("videoUpload");
    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    console.log(data);
}