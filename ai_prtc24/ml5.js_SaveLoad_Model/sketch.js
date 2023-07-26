let mobilenet; 
let classifier; 
let video; 
let label = 'loading model'; 
let helloButton; 
let iloveyouButton; 
let trainButton;

function modelReady () {
  console.log('Model is ready!!!');     
  classifier.load('./model.json', customModelReady);
}

function customModelReady() {
  console.log('Custom Model is ready!!!'); 
  label = 'model ready';     
  classifier.classify(gotResults);
  
}

function videoReady() {
  console.log('Video is ready!!!');
}

// function whileTraining(loss) { 
//   if (loss == null) {
//     console.log('Training Complete');
//     classifier.classify(gotResults); 
//   } else {
//     console.log(loss);
//   }
//}

function gotResults(error, result) { 
  if (error) {
    console.error(error); 
  } else {
    label = result[0].label;
    classifier.classify(gotResults);
  }
}

function setup() {
  createCanvas(640, 550); 
  video = createCapture(VIDEO); 
  video.hide(); 
  background(0); 
  mobilenet = ml5. featureExtractor('MobileNet', modelReady); 
  classifier = mobilenet.classification(video, videoReady);
  

  // helloButton = createButton('hello'); 
  // helloButton.mousePressed(function() {
  //   classifier.addImage('Hello'); 
  // });

  // iloveyouButton = createButton('iloveyou'); 
  // iloveyouButton.mousePressed(function() {
  //   classifier.addImage('I Love You'); 
  // });

  // trainButton = createButton('train');
  // trainButton.mousePressed(function() {
  //   classifier.train(whileTraining);
  // });
  
  // saveButton = createButton('save');
  // saveButton.mousePressed(function(){
  //   classifier.save();                                    
  // });
}
   
function draw() { 
  background(0); 
  image(video, 0, 0, 640, 480); 
  fill(255); 
  textSize(32); 
  text(label, 10, height - 20);
}