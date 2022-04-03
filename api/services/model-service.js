const { exec } = require('child_process');
const fs = require('fs');

const runModel = () => {
    const filePath = '/home/sachini/HA-EEND/egs/librispeech/v1/run.sh';

    return new Promise((resolve, reject) => {
        console.log("[DEBUG]: Model run started");
        exec(filePath, (error, stdout, stderr) => {
            console.log(stdout);
            console.log(stderr);
            if (error) {
                reject(error);
            }

            console.log("[DEBUG]: Model run successful :)");
            resolve();
        });
    });
}

const readFile = (path) => {
    return new Promise((resolve, reject) => {
        fs.readFile(path, "utf-8", (err, buffer) => {
            if (err) {
                reject(err);
            }

            resolve(buffer);
        })

    })
}

const readResults = (file) => {
    const filePath = `/home/sachini/HA-EEND/egs/librispeech/v1/exp/diarize/infer/real/${file.name.split(".")[0]}.h5`;
    const command = `python /home/sachini/HA-EEND/eend/pytorch_backend/read_output.py ${filePath}`
    const speaker1ResultPath = "/home/sachini/HA-EEND/egs/librispeech/v1/exp/diarize/infer/real/output/speaker1"
    const speaker2ResultPath = "/home/sachini/HA-EEND/egs/librispeech/v1/exp/diarize/infer/real/output/speaker2"

    return new Promise((resolve, reject) => {
        console.log("[DEBUG]: File read started");

        exec(command, async (error, stdout, stderr) => {
            console.log(stdout);
            console.log(stderr);
            if (error) {
                reject(error);
            }

            console.log("[DEBUG]: Data converted successfully");

            const speaker1Segments = await readFile(speaker1ResultPath);
            const speaker2Segments = await readFile(speaker2ResultPath);

            const data = {
                speaker1: speaker1Segments.split(" "),
                speaker2: speaker2Segments.split(" ")
            }

            resolve(data);
        });
    });
}


module.exports = {
    runModel,
    readResults
}