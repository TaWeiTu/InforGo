'use strict'

// require modules
var path       = require('path');
var bodyParser = require('body-parser');
var express    = require('express')
var app        = require('express')();
var server     = require('http').Server(app);
var io         = require('socket.io')(server);
var http       = require('http');
var colors     = require('colors/safe');
var rdmString  = require('randomstring');

// variables
var fs = require('fs');
var recordRoot = __dirname + "/saves/" + rdmString.generate(16) + '/';
var fileNum = 0;
var record = "";
fs.mkdir(recordRoot, function (err) {
	if(err) throw err;
});
var port = 9003;
colors.setTheme ({
	setup : ['green', 'underline'],
	info  : ['grey', 'underline'],
	error : ['red', 'underline'],
	title : ['blue', 'bold']
});

app.use('/static', express.static(__dirname + '/static'));

app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', function (req, res) {
	console.log( colors.title("[System] ") + "Get request to '/'" );
	res.sendFile(__dirname + '/main.html', function () {
		res.end();
	});
});

function getTime () {
	let now = new Date();
	return now.toLocaleDateString() + ' ' + now.toLocaleTimeString()	
}

// Bingo part

// variables
var playerList = [];
var waitingList = [];
var onlineList = [];
var player = 1;
var playing = false;
var stat_1D = []; //creating 1D array
var stat_3D = []; //creating 3D array
for (let i = 0; i < 4; i++){
	stat_3D[i] = [];
	for (let j = 0; j < 4; j++) stat_3D[i][j] = [0, 0, 0, 0];
}

// initailize the stat array (turn the placable box into "3",the other to "0")
function initialize () {
	// set stat_1D to 0 or 3
	for (let i = 0; i < 64; i++){
	    if ( i % 4 == 0 ) stat_1D[i] = 3;
	    else stat_1D[i] = 0;
	}
}
initialize();// Important. If removed, clients might make box's material into null.

// check if the game is over.
function checkWinner ( id ) {
	// convert stat to 3D array
	for (let i = 0; i < 64; i++) stat_3D[i % 4][Math.floor(i / 4) % 4][Math.floor(i / 16)] = stat_1D[i];

	// check each line if there are 4 cube with same color in a line
	let x = id % 4;
	let y = Math.floor(id / 4) % 4;
	let z = Math.floor(id / 16);
	let value = 1;
	let sum = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ];
	for (let i = 0; i < 4; i++, value *= 2){
		let point = [stat_3D[x][y][i], stat_3D[x][i][z], stat_3D[i][y][z], stat_3D[x][i][i], stat_3D[x][i][3-i], stat_3D[i][y][i], stat_3D[i][y][3-i], stat_3D[i][i][z], stat_3D[i][3-i][z], stat_3D[i][i][i], stat_3D[i][i][3-i], stat_3D[3-i][i][i], stat_3D[3-i][i][3-i] ];
		for (let line = 0; line < point.length; line++){
			if (point[line] == 1) sum[line] += value;
			if (point[line] == 2) sum[line] -= value;
		}
	}

	// return 1 if there is a line with value "15", ortherwise return 2 if got value"-15". if not, return 0.
	for (let i = 0; i < sum.length; i++){
		if (sum[i] == 15) return 1;
		if (sum[i] ==-15) return 2;
	}
	// return 0 if the game could keep going on.
	for (let i = 0; i < 64; i++){
		if(stat_1D[i] == 3 ) return 0;
	}
	// return 3 if there is no empty place.
	return 3;
}

function tryStart () {
	if (waitingList.length < 2 || playing) return;
	//renew player list
	playerList[0] = waitingList.shift(1);
	playerList[1] = waitingList.shift(1);
	//setup variables and stats
	record = "";
	playing = true;
	player = 1;
	initialize();
	//send messages to clients
	io.emit('restart');
	io.emit('refreshState', stat_1D, player);
	io.to( playerList[0]["id"] ).emit( 'youare', 1 );
	io.to( playerList[1]["id"] ).emit( 'youare', 2 );
	console.log(colors.title("[Bingo] ") + "Game start!");
}

function gameOver (gameInfo){
	for (let i = 0; i < 64; i++) if (stat_1D[i] == 3) stat_1D[i] = 0;
	io.emit('gameOver', gameInfo);
	io.emit('refreshState', stat_1D, 3);
	playing = false;
	player = 3;
	setTimeout( tryStart, 10000);
	if (gameInfo.endWay != 0) writeRecord();
	console.log(colors.title("[Bingo] ")+colors.info("Game over."));
}

function appendRecord ( num ){
	record += num % 4 + ' ';
	record += Math.floor(num / 16)  + ' ';
	record += Math.floor(num / 4) % 4 + ' ';
	record += '\n';
}

function downReq (player_id, num) {
	// filter
	if (!playing) return;
	if (player_id != playerList[player - 1]["id"]) return;
	if (stat_1D[num] != 3) return;
	// update stat_1D and record
	stat_1D[num] = player;
	if ( (num % 4) != 3 ) stat_1D[num + 1] = 3;
	appendRecord(num);

	// check winner
	let winnerId = checkWinner(num);
	if (winnerId == 3) gameOver({'endWay': 3});
	if (winnerId == 0){
		player = (player == 1) ? 2 : 1;
		io.emit('refreshState', stat_1D, player);
	}
	else {
		// check if game is over
		gameOver({
			'endWay': winnerId,
			'winnerId': winnerId,
			'winnerName': playerList[winnerId - 1].name
		});
		waitingList.push(playerList[1]);
		waitingList.push(playerList[0]);
	}
}

function writeRecord (){
	record += "-1 -1 -1\n"
	let recordPath = recordRoot + fileNum.toString();
	fs.writeFile(recordPath, record, function (err) {
		if(err) throw err;
		record = "";
	});
    fileNum++;
}


io.sockets.on('connection', function(socket){

	//when somebody connecting
	var name;
	console.log( colors.title("[Bingo] ") + colors.info("Someone joined") );
	socket.emit('loginHint', playing);
	if (playing) socket.emit('refreshState', stat_1D, player);
	else socket.emit('refreshState', stat_1D, 3);

	//when receiving login request
	socket.on('loginreq', function (name, join){
		// escape
		if (typeof(name) == 'string'){
			name = name.replace(/&/g,"&amp;");
			name = name.replace(/</g,"&lt;");
			name = name.replace(/>/g,"&gt;");
		}
		if (join) waitingList.push({ name: name, id: socket.id });
		onlineList.push({ name: name, id: socket.id });
		tryStart();
	})
	socket.on('downReq', function ( num ) {
		if (typeof(num) != 'number') return;
		downReq(socket.id, num);
		
	});
	socket.on('disconnect',function(){
		console.log(colors.title("[Bingo] ") + colors.info("Someone has discennected"));

		// remove info from lists
		for (let i = 0; i < onlineList.length; i++){
			if (onlineList[i]["id"] == socket.id){
				onlineList.splice(i, 1);
				break;
			}
		}
		for (let i = 0; i < waitingList.length; i++){
			if (waitingList[i]["id"] == socket.id) {
				waitingList.splice(i, 1);
				break;
			}
		}

		// if player leave
		if (playing){
			if (playerList[0]["id"] == socket.id){
				gameOver({ 'endWay': 0, 'winnerId': 2, 'winnerName':playerList[1].name })
				waitingList.push(playerList[1]);
			}
			if (playerList[1]["id"] == socket.id){
				gameOver({ 'endWay': 0, 'winnerId': 1, 'winnerName':playerList[0].name });
				waitingList.push(playerList[0]);
			}
		}
	});
});

server.listen(port, function () {
	console.log(colors.setup("Server is running at port " + port));
});




//Notes :
// set player to 3 while playing is false
// edit Line124 : player = 3;
// to optimize Line 173
// may affect : wrong index with player-1

// sockets document
// 'gameOver': { endWay[, winnerId, winnerName ]} , 
//		endWay: 0 = leave, 1,2 = player wins, 3 = draw
