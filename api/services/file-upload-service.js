const fs = require('fs')

const uploadAndSaveFile = (file) => {
    uploadPath = '/home/sachini/HA-EEND/egs/librispeech/v1/data/simu/wav/real/' + file.name;

    return new Promise((resolve, reject) => {
        file.mv(uploadPath, err => {
            if (err)
                reject(err);

            console.log("[DEBUG]: Wav file moved")
            resolve();
        });
    });
}

const updateWavScp = (file) => {
    filePath = '/home/sachini/HA-EEND/egs/librispeech/v1/data/simu/data/real/wav.scp'
    // TODO: update to support general case
    content = `${file.name.split(".")[0]} /home/sachini/HA-EEND/egs/librispeech/v1/data/simu/wav/real/${file.name}`

    return new Promise((resolve, reject) => {
        fs.writeFile(filePath, content, err => {
            if (err) {
                reject(err);
            }

            console.log("[DEBUG]: Scp file updated")
            resolve();
        })
    });
}  

module.exports = {
    uploadAndSaveFile,
    updateWavScp
}