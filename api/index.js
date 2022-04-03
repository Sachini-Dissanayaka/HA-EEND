const express = require("express");
const fileUpload = require("express-fileupload");
const { uploadAndSaveFile, updateWavScp } = require("./services/file-upload-service");
const { runModel, readResults } = require("./services/model-service");

const app = express();

app.use(fileUpload());

app.get("/", (req, res) => {
    res.send("Welcome to API Version 1.0.0");
});

app.post('/ha-eend', async (req, res) => {
    console.log("[DEBUG]: Request Hit")
    if (!req.files || Object.keys(req.files).length === 0) {
        return res.status(400).send('No files were uploaded.');
    }

    try {
        const file = req.files.wavFile;
        await uploadAndSaveFile(file);
        await updateWavScp(file);
        await runModel();
        await readResults(file);
        return res.status(200).send('Success!');
    } catch (err) {
        return res.status(500).send('Something went wrong! ' + err);
    }
});

const PORT = 3000;

app.listen(PORT, () => {
    console.log("[DEBUG]: Server is listening on port : " + PORT);
});

