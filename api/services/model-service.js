const { exec } = require('child_process');
const fs = require('fs');
const { Readable } = require('stream')

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

const readResults = (file) => {
    const filePath = `/home/sachini/HA-EEND/egs/librispeech/v1/exp/diarize/infer/real/${file.name.split(".")[0]}.h5`;
    const command = `python /home/sachini/HA-EEND/eend/pytorch_backend/read_output.py ${filePath}`

    return new Promise((resolve, reject) => {
        console.log("[DEBUG]: File read started");

        exec(command, (error, stdout, stderr) => {
            console.log(stdout);
            console.log(stderr);
            if (error) {
                reject(error);
            }

            console.log("[DEBUG]: Data converted successfully");
            resolve();
        });

        // fs.readFile(filePath, "base64", (err, buffer) => {
        //     if (err) {
        //         reject(err);
        //     }

        //     console.log("[DEBUG]: "+ buffer.toString());
        //     resolve();
        // })
    });
}


module.exports = {
    runModel,
    readResults
}