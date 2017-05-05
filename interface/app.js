'use strict'

// require modules
var express    = require('express')
var app        = require('express')()
var server     = require('http').Server(app)
var io         = require('socket.io')(server)
var http       = require('http')
var bingo      = require('./static/js/bingo_server.js')
var config     = require("./config.json")
var DEBUG = true

var port = config.port

bingo.init({ 
	'io':io,
	'config':config,
	'DEBUG':DEBUG
});

app.use('/assets', express.static(__dirname + '/assets'))
app.use('/static', express.static(__dirname + '/static'))
app.get('/', function(req, res){
	res.sendFile(__dirname + '/pages/clients.html', function(){
		res.end()
	})
})
app.get('/simulate', function(req, res){
	res.sendFile(__dirname + '/pages/simulate.html', function(){
		res.end()
	})
})



//start.js

io.sockets.on('connection', function(socket){
	
	// initialize
	let player = new bingo.Player(socket)
	console.log("[Bingo] Someone connented.")
	socket.emit('refreshRoomInfo', { 'list':bingo.getSimpleRoomList() })

	// socket events
	socket.on('createRoomReq', function(data){
		if (!data.playerName || !data.roomName || !data.mode) return
		if (!player.name) player.name = escape(data.playerName.toString())
		data.roomName = escape(data.roomName.toString())
		createRoom(player, data.roomName, data.mode)
	})

	socket.on('joinRoomReq', function(name, rid){
		if(!name || !rid || rid == player.rid) return
		if(!player.name) player.name = escape(name.toString())
		joinRoom(player, rid)
	})

	socket.on('joinGameReq', function(){
		if(!player.rid) return
		let room = bingo.getRoomByRid(player.rid)
		if(!room) return
		player.joinGame(room)
	})

	socket.on('disconnect', function(){
		bingo.playerDisconnect(player)
	})

	socket.on('downReq',function(num){
		if(player.rid) bingo.getRoomByRid(player.rid).downReq(player.id, num)
	})

	// for admin
	socket.on('removeRoom', function(rid, passwd){
		if (passwd != config.passwd) return
		bingo.getRoomByRid(rid).announce('message', {'message':'Admin removed the room.', 'msgId':bingo.randomString(8)})
		bingo.getRoomByRid(rid).removeRoom(bingo.roomList)
		io.emit('refreshRoomInfo', { 'list':bingo.getSimpleRoomList() })		
	})
	socket.on('check', function(params){
		if (params == 'roomList') console.log(bingo.getSimpleRoomList())
		if (params == 'id') console.log(player.id)
	})
})

function joinRoom(player, rid){
	player.joinRoom(bingo.getRoomByRid(rid))
	player.socket.emit('message', {'message':'Join room successfully.', 'msgId':bingo.randomString(8)})
}

function createRoom(player, roomName, mode){
	if (mode != 'pvp' && mode != 'com') return
	let newRoom = new bingo.Room(roomName, mode)
	bingo.roomList.push(newRoom)
	player.joinRoom(newRoom);
	player.socket.emit('createRoomRes', {'status':'success','rid':newRoom.rid})
	io.emit('refreshRoomInfo', { 'list':bingo.getSimpleRoomList() })
	if (DEBUG) console.log("[Bingo] Player", player.name, "created room", roomName)
}

function searchPlayerInRoomList(socket){
	for (let i = 0; i < bingo.roomList.length; ++i){
		for (let j = 0; j < audList.length; ++j) if (audList[i] == socket) return bingo.roomList[i]
		for (let j = 0; j < playerList.length; ++j) if (playerList[i] == socket) return bingo.roomList[i]
	}
	return null
}

function escape(str){
	return str.replace(/&/g, "&amp").replace(/</g, "&lt").replace(/>/g, "&gt")
}

function jizz(){
	for (let i=0;i<5;i++){
		let a = Math.floor(Math.random()*10+1)
		let b = Math.floor(Math.random()*10+1)
		console.log(Math.min(a, b), Math.max(a, b), Math.floor(Math.random()*50))
	}
	for (var i = 0; i < 8; i++) {
		console.log(Math.floor(Math.random()*10+1),Math.floor(Math.random()*20)+1)
	};
}

server.listen(port, function(){
	console.log("Server is running at port", port)
})
