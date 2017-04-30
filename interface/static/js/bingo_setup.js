var socket = io.connect();

var info = document.getElementById("info")
var gameBoard = document.getElementById("gameBoard")
var messageBox = document.getElementById("messageBox")

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
    cubes[i].position.x += ((i - i % 16) / 16) + 1;
    cubes[i].position.y += i % 4 + 0.4;
    cubes[i].position.z += (i % 16 - i % 4)/4 + 1;
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
var turn = 3;

// set up raycaster
var raycaster = new THREE.Raycaster();
var firstVisibleObject, selected, intersects;

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
                if(selfId == 1)selected.material = TransBlue;
                if(selfId == 2)selected.material = TransRed;
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
        socket.emit('downReq', firstVisibleObject.Num);
    }
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

function renderRefresh(gameStat){
    for(let i = 0; i < 64; ++i){
        cubes[i].material = color[gameStat[i]];
        cubes[i].situation = gameStat[i];
    }
}


socket.on('restart', function(){
    p1Icon1.material = colorBlue;
    p1Icon2.material = colorBlue;
    p2Icon1.material = colorRed;
    p2Icon2.material = colorRed;
    p1Icon1.geometry = iconBox_small;
    p1Icon2.geometry = iconBox_small;
    p2Icon1.geometry = iconBox_small;
    p2Icon2.geometry = iconBox_small;
    renderer.setClearColor(0xbbbbbb);
})
socket.on('playerAnnounce',function(playerNum){
    selfId = playerNum;
    if (selfId == 1){
        alert("遊戲開始~~藍方先下子");
        renderer.setClearColor(0xbbbbdd);
    }
    if (selfId == 2){
        alert("遊戲開始~~紅方請稍候");
        renderer.setClearColor(0xddbbbb);
    }
})
socket.on('refreshState',function(data){
    renderRefresh(data.stat);
    turn = data.turn;
})
socket.on('gameOver',function(gameInfo){
    if (gameInfo.endWay == 3){
        alert("此局平手，10秒後開啟新局");
    }
    if (gameInfo.endWay == 0){
        if (gameInfo.winnerId == 1) alert("紅方離開遊戲，藍方 " + gameInfo.winnerName + " 勝利~ 10秒後等候另外兩位參賽者開啟新局");
        if (gameInfo.winnerId == 2) alert("藍方離開遊戲，紅方 " + gameInfo.winnerName + " 勝利~ 10秒後等候另外兩位參賽者開啟新局");
    }
    else {
        if (gameInfo.winnerId == 1) alert("藍方 " + gameInfo.winnerName + " 勝利~ 10秒後等候另外兩位參賽者開啟新局");
        if (gameInfo.winnerId == 2) alert("紅方 " + gameInfo.winnerName + " 勝利~ 10秒後等候另外兩位參賽者開啟新局");
    }
    if (gameInfo.endWay != 3){
        p1Icon1.material = color[gameInfo.winnerId];
        p1Icon2.material = color[gameInfo.winnerId];
        p2Icon1.material = color[gameInfo.winnerId];
        p2Icon2.material = color[gameInfo.winnerId];
    }
    p1Icon1.geometry = iconBox_big;
});
    p1Icon2.geometry = iconBox_big;
    p2Icon1.geometry = iconBox_big;
    p2Icon2.geometry = iconBox_big;
    selfId = null;

// declare room variables
var nowRoomId = null;
var ridList = [];


// socket events
socket.on('refreshRoomInfo',function(data){
    refreshRoomInfo(data.list);
})
socket.on('message', function(data){
    spawnMessage(data.message,data.msgId);
})
socket.on('joinRoomRes', function(data){
    if(data.status == 'success'){
        nowRoomId = data.rid
        document.getElementById('joinButton').disabled = !data.joinable
    }
})
socket.on('joinGameRes', function(data){
    if(data.status == 'success'){
        document.getElementById('joinButton').disabled = true
        spawnMessage('Join game success.')
    }
    if(data.status == 'failed'){
        spawnMessage('Join game failed.')
    }
})
socket.on('createRoomRes', function(data){
    if(data.status == 'success') socket.emit('joinGameReq')
})

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

// create room request
function createRoom(){
    if (!document.getElementById('name').value){
        spawnMessage("Name can't be null.", Math.random().toString(36).substring(8))
        return
    }
    let roomName = prompt("請輸入房間名稱")
    if(!roomName){
        if (roomName == '') spawnMessage("Room name can't be null.", Math.random().toString(36).substring(8))
        return
    }
    document.getElementById('name').disabled = true;
    let mode = 'pvp';
    if (document.getElementById('mode').checked) mode = 'com';
    socket.emit('createRoomReq',{ 'playerName':document.getElementById('name').value, 'roomName':roomName, 'mode':mode });
}

// refresh sidebar room info
function refreshRoomInfo(list){
    document.getElementById('room-info').innerHTML = "";
    for (let i = 0; i < list.length; i++){
        let flag = true
        for (let j = 0; j < ridList.length; j++){
            if (list[i].rid == ridList[j]){
                document.getElementById('room-info').innerHTML += "<tr class=\"roomList\" id=\'{0}\' onclick='joinRoom(\"{0}\")'><td>{1}</td><td>{2}</td><td>{3}</td></tr>".format(list[i].rid, list[i].name, list[i].mode, list[i].playing);
                flag = false
                break;
            }
        }
        if (flag) document.getElementById('room-info').innerHTML += "<tr class=\"roomList animated fadeInUp\" id=\'{0}\' onclick='joinRoom(\"{0}\")'><td>{1}</td><td>{2}</td><td>{3}</td></tr>".format(list[i].rid, list[i].name, list[i].mode, list[i].playing);
    }
    ridList = []
    for(let i = 0; i < list.length; i++) ridList.push(list[i].rid);
}

function joinRoom(rid){
    if (!document.getElementById('name').value){
        spawnMessage("Name can't be null.", Math.random().toString(36).substring(8))
        return
    }
    document.getElementById('name').disabled = true;
    if (nowRoomId != rid){
        socket.emit('joinRoomReq', document.getElementById('name').value, rid);
    }
}

function joinGame(){
    socket.emit('joinGameReq')
}

// string format function
String.prototype.format = function(){
    let s = this, i = arguments.length;
    while(i--)s = s.replace(new RegExp('\\{'+i+'\\}', 'gm'), arguments[i]);
    return s;
};