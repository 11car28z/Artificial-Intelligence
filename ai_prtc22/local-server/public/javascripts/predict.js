$("#image-selector").change( function () {
    let reader = new FileReader();
    reader.onload = () => {
        let dataURL = reader.result;
        $('#selected-image').attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});

$("#model-selector").change(function () {
    loadModel($("#model-selector").val());
});

let model;
async function loadModel(name) {
    $(".progress-bar").show();
    model = undefined;
    MODEL_PATH = `/tfjs-models/${name}/model.json`;
    model = await tf.loadLayersModel(MODEL_PATH);
    console.log("Model loaded!!")
    $(".progress-bar").hide();
}

$("#predict-button").click(async () => {
    let image = $('#selected-image').get(0);
    let modelName = $("#model-selector").val();
    console.log('modelName:\n', modelName);
    let tensor = preprocessImage(image, modelName);

    let predictions = await model.predict(tensor).data();
    console.log('predictions:\n', predictions);

    let top5 = Array.from(predictions)
        .map(function (p, i) {
            return {
                probability: p,
                className: DOGBREED_CLASSES[i]
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;
        }).slice(0, 5);

    $("#prediction-list").empty();
    top5.forEach(function (p) {
        $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`)
    });
});

function preprocessImage(image, modelName) {
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat();

    if (modelName === undefined) {
        return tensor.expandDims();
    }
    else if (modelName === "VGG16") {
        let meanImageNetRGB = tf.tensor1d([123.68, 116.779, 103.939]);
        return tensor.sub(meanImageNetRGB)
            .reverse(2)
            .expandDims();
    }
    else if (modelName === "MobileNetV2") {
        let offset = tf.scalar(127.5);
        return tensor.sub(offset)
            .div(offset)
            .expandDims();
    }
    else {
        throw new Error("Unknown model name");
    }
};
