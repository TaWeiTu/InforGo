'use strict'

// require modules
var express    = require('express')
var app        = require('express')()
var server     = require('http').Server(app)
var io         = require('socket.io')(server)
var http       = require('http')
var bingo      = require('./js/bingo.js')
var config     = require("./config.json")

var port = config.port

app.use('/assets', express.static(__dirname + '/assets'))
app.use('/static', express.static(__dirname + '/static'))
app.get('/lobby', function(req, res){
	res.sendFile(__dirname + '/pages/clients.html', function(){
		res.end()
	})
})
app.get('/game', function(req, res){
	res.sendFile(__dirname + '/pages/elements.html', function(){
		res.end()
	})
})
var roomList = []
//start.js

io.sockets.on('connection', function(socket){
	let player = new bingo.Player(socket)
	console.log("someone in")
	socket.emit('refreshRoomInfo', getSimpleRoomList(roomList))
	socket.on('createRoomReq', (name, mode) => {
		if(mode != 'pvp' && mode != 'com')return
		console.log('get createRoomReq with mode',mode)
		console.log(name)
		if(!name)return
		else name = escape(name.toString())
		console.log(name)
		if(typeof player.name === 'undefined')player.name = name
		else name = player.name
		let newRoom = new bingo.Room(name, mode)
		roomList.push(newRoom)
		newRoom.playerList.push(player);
		socket.emit('message', {'message':'Room create sucess.', 'msgId':bingo.randomString(8)})

		io.emit('refreshRoomInfo',getSimpleRoomList(roomList))
	})
	socket.on('joinRoomReq', (name, rid) => {
		if(!name)return
		else name = escape(name.toString())
		if(typeof player.name === 'undefined')player.name = name
		else name = player.name
		let room = getRoomByRid(rid)
		room.audList.push(player)
		socket.emit('message', {'message':'successful join room.'})
	})
	socket.on('audBecomePlayer', rid => {
		if(!player.name)return
		let room = getRoomByRid(rid)
		if(room.playerList.length == 2){
			socket.emit('message', {'message':'failed', 'detail':'player already full.', 'msgId': bingo.randomString(8)})
			return
		}
		for(let i=0;i<room.audList.length;++i){
			if(room.audList[i] == player){
				room.audList.splice(i,1)
				room.playerList.push(player)
				socket.emit('message', {'message':'success become player', 'msgId':bingo.randomString(8)})
				return
			}
		}
		socket.emit('message',{'message':'failed become player', 'detail':'can\'t find such player in list.', 'msgId': bingo.randomString(8)})
	})
	socket.on('removeRoom', rid => {
		console.log('Get remove room request')
		getRoomByRid(rid).removeRoom(roomList,io)
		console.log('Remove sucess.')
		console.log('Room list',roomList)
		io.emit('refreshRoomInfo',getSimpleRoomList(roomList))
	})
})

function getRoomByRid(rid){
	for(let i=0;i<roomList.length;i++){
		if(roomList[i].rid == rid)return roomList[i]
	}
	throw "Room not found"
}

function searchPlayerInRoomList(socket){
	for(let i=0;i<roomList.length;i++){
		for(let j=0;j<audList.length;j++)if(audList[i] == socket)return roomList[i]
		for(let j=0;j<playerList.length;j++)if(playerList[i] == socket)return roomList[i]
	}
	return null
}

function getSimpleRoomList(list){
	let simpleList = []
	for(let i=0;i<list.length;i++){
		simpleList.push({
			'rid': list[i].rid,
			'name': list[i].name,
			'mode': list[i].mode,
			'playing': list[i].playing
		})
	}
	return simpleList
}

function escape(str){
	return str
	// return str.replace(/&/g, "&amp").replace(/</g, "&lt").replace(/>/g, "&gt")
}

server.listen(port, function(){
	console.log("Server is running at port",port)
})
