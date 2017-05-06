var info = document.getElementById("info")
var gameBoard = document.getElementById("gameBoard")
var messageBox = document.getElementById("messageBox")
var recInput = document.getElementById("recInput")
var autorun = document.getElementById("auto")
var startButton = document.getElementById("startButton")

// set up the renderer
var scene = new THREE.Scene();
// var camera = new THREE.PerspectiveCamera( 70, window.innerWidth*0.75/window.innerHeight, 1, 1000 );
var camera = new THREE.PerspectiveCamera(70, 1.2, 1, 1000);
var renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth*0.75, window.innerHeight );
gameBoard.appendChild(renderer.domElement);
renderer.setClearColor(0xbbbbbb);

// set up material & geometry
var basicGeometry = new THREE.BoxGeometry(5, 0.1, 5);
var iconBox_small = new THREE.BoxGeometry(0.5, 0.5, 0.5);
var iconBox_big = new THREE.BoxGeometry(0.7, 0.7, 0.7);
var colorInvis = new THREE.MeshLambertMaterial({opacity: 0, transparent: true});
var colorClickable = new THREE.MeshLambertMaterial({color: 0x555555, opacity: 0.3, transparent: true});
var colorBlue  = new THREE.MeshLambertMaterial({color: 0x0000ff});
var TransBlue  = new THREE.MeshLambertMaterial({color: 0x0000ff, opacity: 0.6, transparent: true});
var colorRed   = new THREE.MeshLambertMaterial({color: 0xff0000});
var TransRed   = new THREE.MeshLambertMaterial({color: 0xff0000, opacity: 0.6, transparent: true});
var colorGray  = new THREE.MeshLambertMaterial({color: 0x222222});
var redShine   = new THREE.MeshLambertMaterial({color: 0xff0000, opacity: 0, transparent: true});
var blueShine  = new THREE.MeshLambertMaterial({color: 0x0000ff, opacity: 0, transparent: true});
var color = [colorInvis , colorBlue , colorRed , colorClickable , TransBlue , TransRed];

// set up basic(called cube)
var cube = new THREE.Mesh(basicGeometry, colorGray);
scene.add(cube);
cube.position.x = 2.5;
cube.position.z = 2.5;

// set up Icon
var p1Icon1 = new THREE.Mesh(iconBox_small, colorBlue);
var p1Icon2 = new THREE.Mesh(iconBox_small, colorBlue);
var p2Icon1 = new THREE.Mesh(iconBox_small, colorRed);
var p2Icon2 = new THREE.Mesh(iconBox_small, colorRed);
scene.add(p1Icon1);
scene.add(p1Icon2);
scene.add(p2Icon1);
scene.add(p2Icon2);
p1Icon1.position.x = -0.5;
p1Icon1.position.z = -0.5;
p1Icon2.position.x = 5.5;
p1Icon2.position.z = 5.5;
p2Icon1.position.x = -0.5;
p2Icon1.position.z = 5.5;
p2Icon2.position.x = 5.5;
p2Icon2.position.z = -0.5;

// set up cubes
var cubes = [];
for(let i = 0; i < 64; ++i){
    cubes[i] = new THREE.Mesh(iconBox_small, colorInvis);
    scene.add(cubes[i]);
    cubes[i].situation = 0;
    cubes[i].Num = i;
    cubes[i].position.x += ((i - i % 16) / 16) + 1;
    cubes[i].position.y += i % 4 + 0.4;
    cubes[i].position.z += (i % 16 - i % 4)/4 + 1;
    // if(i % 4 == 0){
    //     cubes[i].material = colorClickable;
    //     cubes[i].situation = 3;
    // }
}

// set up light
var ambientLight = new THREE.AmbientLight(0x888888 ,0.5);
scene.add(ambientLight);
var directionalLight = new THREE.DirectionalLight(0xcccccc);
directionalLight.position.set(1, 2, 2).normalize();
scene.add(directionalLight);

// camera controll
var manualControl = false;
var longitude = 0;
var latitude = 0;
var savedX;
var savedY;
var savedLongitude;
var savedLatitude;
var scale = 60;
var mouse = new THREE.Vector2();
camera.target = new THREE.Vector3(0, 0, 0);

// variable declar
var selfId;
var turn = 3;

// set up raycaster
var raycaster = new THREE.Raycaster();
var firstVisibleObject, selected, intersects;
var t = 0, savedOpacity;
render();

function render(){

    requestAnimationFrame(render);
    // limiting latitude from 0 to 85
    latitude = Math.max(-85, Math.min(0, latitude));
    // maintain the window in right size
    renderer.setSize( window.innerWidth*0.75, window.innerHeight);

    //set camera
    camera.target.x = 500 * Math.sin(THREE.Math.degToRad(90 - latitude)) * Math.cos(THREE.Math.degToRad(longitude));
    camera.target.y = 500 * Math.cos(THREE.Math.degToRad(90 - latitude));
    camera.target.z = 500 * Math.sin(THREE.Math.degToRad(90 - latitude)) * Math.sin(THREE.Math.degToRad(longitude));
    camera.lookAt(camera.target);
    camera.position.x = -camera.target.x / scale +2.5;
    camera.position.y = -camera.target.y / scale + 1;
    camera.position.z = -camera.target.z / scale +2.5;

    //rotate icon
    if(turn == 1){
        p1Icon1.rotation.y += 0.1;
        p1Icon2.rotation.y += 0.1;
    }
    else if(turn==2){
        p2Icon1.rotation.y += 0.1;
        p2Icon2.rotation.y += 0.1;
    }
    else if(turn==3){
        p1Icon1.rotation.y += 0.1;
        p1Icon2.rotation.y += 0.1;
        p2Icon1.rotation.y += 0.1;
        p2Icon2.rotation.y += 0.1;
    }
    t+=0.04
    savedOpacity = Math.sin(t) / 4 + 0.75
    redShine.opacity = savedOpacity
    blueShine.opacity = savedOpacity
    renderer.render(scene, camera);
}

var step = 0, splitedInput, record, steps, run, interval = 3.5, keeprun = true


function loadInput(){

    // initialize
    splitedInput = recInput.value.split('\n')
    for (let i = 0; i < splitedInput.length;){
        for(let j = 0; j < splitedInput[i].length;){
            if (splitedInput[i][j] == ' '){
                splitedInput[i] = splitedInput[i].splice(j, 1)
            }
            else j++
        }
        if(splitedInput[i].length != 2){
            splitedInput.splice(i, 1)
        }
        else i++
    }
    record = []
    steps = []
    for(let i = 0; i < splitedInput.length; i++){
        record.push([])
    }
    // load input
    for(let i = 0; i < 64; i++){
        record[0].push(0)
    }
    steps[0] = convert(record[0], splitedInput[0])
    record[0][steps[0]] = 1
    for(let i = 1; i < splitedInput.length; i++){
        record[i] = record[i-1].slice()
        steps[i] = convert(record[i], splitedInput[i])
        record[i][steps[i]] = (i % 2 == 0)? 1:2
    }
}

function convert(stat, inp){
    let rowId = parseInt(inp[0])*4 + parseInt(inp[1])*16
    for (let i = 0; i < 4; i++){
        if (stat[rowId + i] == 0){
            return rowId + i
        }
    }
    console.log("[Bingo] WTFFFFFFFFFFFFF Full row checked!!")
    return
}

function reset(){
    for (let i = 0; i < 64; i++) cubes[i].material = colorInvis
    record = []
    recInput.disabled = false
    startButton.className = "button special"
    keeprun = false
}

function start(){
    if  (startButton.disabled) return
    step = 0
    recInput.disabled = true
    startButton.className = "button special disabled"
    loadInput()
    draw(record[step],steps[step])
    setTimeout(run, interval*1000)
}

function run(){
    if (keeprun){
        setTimeout(run,interval*1000)
        if(autorun.checked && step < splitedInput.length-1){
            next()
        }
    }
    else keeprun = true
}

function next(){
    if (step == splitedInput.length-1 ){
        spawnMessage("It's finsl step!!")
        return
    }
    step++
    draw(record[step],steps[step])
}

function prev(){
    if (step == 0){
        spawnMessage("It's first step!!")
        return
    }
    step--
    draw(record[step],steps[step])
}



function draw(stat, last){
    for(let i = 0; i < 64; ++i){
        cubes[i].material = color[stat[i]];
        cubes[i].situation = stat[i];
    }
    if(stat[last] == 1) cubes[last].material = blueShine
    else cubes[last].material = redShine
}

function renderRefresh(gameStat){
    for(let i = 0; i < 64; ++i){
        cubes[i].material = color[gameStat.stat[i]];
        cubes[i].situation = gameStat.stat[i];
    }
    if(typeof(gameStat.last) != 'undefined'){
        if(gameStat.stat[gameStat.last] == 1) cubes[gameStat.last].material = blueShine
        else cubes[gameStat.last].material = redShine
    }
}


// listeners
gameBoard.addEventListener("mousedown", onDocumentMouseDown, false);
gameBoard.addEventListener("mousemove", onDocumentMouseMove, false);
gameBoard.addEventListener("mouseup", onDocumentMouseUp, false);
gameBoard.addEventListener("wheel", onDocumentWheel, false);



function onDocumentMouseDown(event){
    manualControl = true;

    savedX = event.clientX;
    savedY = event.clientY;
    savedLongitude = longitude;
    savedLatitude = latitude;
}

function onDocumentMouseMove(event){

    let gameBoardWidth = parseFloat(window.getComputedStyle(gameBoard.children[0]).width)
    let gameBoardHeight = parseFloat(window.getComputedStyle(gameBoard.children[0]).height)
    if(manualControl){
        longitude = -(savedX - event.clientX) * 0.1 + savedLongitude;
        latitude = -(event.clientY - savedY) * 0.1 + savedLatitude;
    }
    mouse.x = ( (event.clientX - gameBoard.offsetLeft) / gameBoardWidth) * 2 - 1;
    mouse.y = -(event.clientY / gameBoardHeight) * 2 + 1;
}

function onDocumentMouseUp(event){
    manualControl = false;
}
function onDocumentWheel(event){
    scale += event.deltaY / 25;
    scale = Math.max(scale, 30);
    scale = Math.min(scale, 80);
}

function spawnMessage(text, id){
    if(!id) id = Math.random().toString(36).substring(8)
    let textBox = document.createElement('div');
    textBox.id = id;
    textBox.className = 'messageBoxText fadeInUp animated';
    textBox.innerHTML = text;
    messageBox.appendChild(textBox);
    setTimeout(function(){
        document.getElementById(id).className += ' fadeOutUp';
    },5000);
    setTimeout(function(){
        messageBox.removeChild(textBox);
    },6000);
}

String.prototype.splice = function (i,j){
    return this.slice(0,i) + this.slice(i+j, this.length) 
}