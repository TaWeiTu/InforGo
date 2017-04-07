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
	console.log("someone in")
	socket.emit('refreshRoomInfo', getSimpleRoomList(roomList))
	socket.on('createRoomReq', function(name, mode){
		console.log("get createRoomReq with mode",mode)
		if (!name)return
		if (mode != 'pvp' && mode != 'com') return
		// escape
		name = name.toString()
		name = name.replace(/&/g, "&amp")
		name = name.replace(/</g, "&lt")
		name = name.replace(/>/g, "&gt")
		roomList.push(new bingo.Room(name, mode))
		socket.emit('message',{'message':'Room create sucess.', 'msgId': bingo.randomString(8)})

		io.emit('refreshRoomInfo',getSimpleRoomList(roomList))
	})
	socket.on('removeRoom',function(rid){
		console.log('Get remove room req')
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

server.listen(port, function(){
	console.log("Server is running at port",port)
})
