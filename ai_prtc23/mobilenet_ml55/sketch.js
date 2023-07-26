let mobilenet;
let bird;

function modelReady(){
  console.log('Model is ready!!!');
  mobilenet.predict(bird, gotResults);
}

function gotResults(error, results){
  if(error){
    console.error(error);
  }else{
    console.log(results);
    let label = results[0].label;
    let confidence = results[0].confidence;
    fill(0);
    textSize(64);
    text(label, 10, height - 100);
    createP(label);
    createP(confidence);
  }
}

function imageReady(){
  image(bird, 0, 0, width, height);
}

function setup() {
  createCanvas(0, 0);
  bird = createImg('images/bird.jpg', imageReady);
  //bird.hide();
  background(1);
  
  mobilenet = ml5.imageClassifier('MobileNet', modelReady)
}

function draw() {
  background(220);
}