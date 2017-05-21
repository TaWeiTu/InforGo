var info = document.getElementById("info")
var gameBoard = document.getElementById("gameBoard")
var rec = document.getElementById("record")

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
scene.add( cube );
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
    cubes[i].position.x = 4 - Math.floor(i / 16)
    cubes[i].position.y = i % 4 + 0.4;
    cubes[i].position.z = Math.floor(i % 16 / 4)   + 1;
    if(i % 4 == 0){
        cubes[i].material = colorClickable;
        cubes[i].situation = 3;
    }
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
var turn = 1;

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

    //set raycaster
    raycaster.setFromCamera(mouse, camera);
    intersects = raycaster.intersectObjects(scene.children);

    //check firstVisibleObject
    firstVisibleObject=null;
    for(let k = 0; k < intersects.length; k++){
        if(intersects[k].object.material != colorInvis && intersects[k].object.geometry != basicGeometry){
            firstVisibleObject = intersects[k].object;
            break;
        }
    }

    //player's actions  ///Note: could be edit
    if(firstVisibleObject){
        if(selected != firstVisibleObject){
            if(selected && selected.situation == 3)selected.material = colorClickable;
            selected = firstVisibleObject;
            if(selected.situation == 3){
                if(turn == 1)selected.material = TransBlue;
                if(turn == 2)selected.material = TransRed;
            }
        }
    }
    else{
        if(selected && selected.situation == 3)selected.material = colorClickable;
        selected = null;
    }

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
    t+=0.05
    savedOpacity = Math.sin(t) / 4 + 0.75
    redShine.opacity = savedOpacity
    blueShine.opacity = savedOpacity
    renderer.render(scene, camera);
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

    if(firstVisibleObject && firstVisibleObject.situation == 3){
        down(firstVisibleObject.Num);
    }
    rec.innerHTML = getRecord()
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

var steps = []

function reset(){
    for(let i = 0; i < 64; ++i){
        cubes[i].situation = 0;
        cubes[i].material = color[0]
        if(i % 4 == 0){
            cubes[i].material = colorClickable;
            cubes[i].situation = 3;
        }
    }
    rec.innerHTML = getRecord()
    steps = []
    turn = 1
}
reset()

function prev(){
    steps.pop()
    for(let i = 0; i < 64; ++i){
        cubes[i].situation = 0
        cubes[i].material = color[0]
        if(i % 4 == 0){
            cubes[i].material = colorClickable;
            cubes[i].situation = 3;
        }
    }
    turn = 1
    for(let i = 0; i < steps.length; i++){
        cubes[steps[i]].situation = turn
        cubes[steps[i]].material = color[turn]
        if (steps[i] % 4 != 3){
            cubes[steps[i] + 1].situation = 3
            cubes[steps[i] + 1].material = color[3]
        }
        turn = turn == 1? 2:1
    }
    rec.innerHTML = getRecord()
}

function down(id){
    steps.push(id)
    cubes[id].situation = turn
    cubes[id].material = color[turn]
    if (id % 4 != 3){
        cubes[id + 1].situation = 3
        cubes[id + 1].material = color[3]
    }
    turn = turn == 1? 2:1
    let way = checkWinner(id)
    if(way != 0) gameOver(way)
}

function gameOver(way){
    alert("gameOver")
    for(let i = 0; i < 64; i++){
        if(cubes[i].situation == 3){
            cubes[i].situation = 0
            cubes[i].material = color[0]
        }
    }
}

var stat_3D = []
for (let i = 0; i < 4; ++i){
    this.stat_3D[i] = []
    for (let j = 0; j < 4; ++j) this.stat_3D[i][j] = [0, 0, 0, 0]
}

function checkWinner(id){
    for(let i = 0; i < 64; i++){
        stat_3D[i % 4][Math.floor(i / 4) % 4][Math.floor(i / 16)] = cubes[i].situation
    }
    // check if the game is over.
    let x = id % 4
    let y = Math.floor(id / 4) % 4
    let z = Math.floor(id / 16)
    let value = 1;
    let sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for (let i = 0; i < 4; ++i, value *= 2){
        let point = [
            stat_3D[x][y][i], stat_3D[x][i][z], stat_3D[i][y][z], // 1D
            stat_3D[x][i][i], stat_3D[i][y][i], stat_3D[i][i][z], // 2D
            stat_3D[x][i][3-i],  stat_3D[i][y][3-i], stat_3D[i][3-i][z], //2D_inverse
            stat_3D[i][i][i], stat_3D[i][i][3-i], stat_3D[3-i][i][i], // 3D
            stat_3D[3-i][i][3-i]
        ]
        for (let line = 0; line < point.length; ++line){
            if (point[line] == 1) sum[line] += value
            if (point[line] == 2) sum[line] -= value
        }
    }

    // return 1 if there is a line with value "15", ortherwise return 2 if got value"-15". if not, return 0.
    for (let i = 0; i < sum.length; ++i){
        if(sum[i] == 15) return 1;
        if(sum[i] ==-15) return 2;
    }
    // return 0 if the game could keep going on.
    for (let i = 0; i < 64; ++i){
        if(cubes[i].situation == 3) return 0;
    }
    // return 3 if there is no empty place.
    return 3;
} 

function getRecord(){
    let record = ""
    let n, s
    for(let i = 0; i < 4; i++){
        for(let j = 0; j < 4; j++){
            for(let k = 0; k < 4; k++){
                n = i * 16 + j + k * 4
                if (cubes[n].situation == 1) s = '1'
                else if (cubes[n].situation == 2) s = '-1'
                else if (cubes[n].situation == 0 || cubes[n].situation == 3) s = '0'
                record += s
                if(j != 4 ) record += ' '
            }
        }
        record += '<br>'
    }
    return record
}

function spawnMessage(text, id){
    if(!id) id = Math.random().toString(36).substring(8)
    let textBox = document.createElement('div');
    textBox.id = id;
    console.log(id);
    textBox.className = 'messageBoxText fadeInUp animated';
    textBox.innerHTML = '{0}'.format(text);
    messageBox.appendChild(textBox);
    setTimeout(function(){
        document.getElementById(id).className += ' fadeOutUp';
    },5000);
    setTimeout(function(){
        messageBox.removeChild(textBox);
    },6000);
}

// string format function
String.prototype.format = function(){
    let s = this, i = arguments.length;
    while(i--)s = s.replace(new RegExp('\\{'+i+'\\}', 'gm'), arguments[i]);
    return s;
};
