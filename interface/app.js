'use strict'

// require modules
var express    = require('express')
var app        = require('express')()
var server     = require('http').Server(app)
var io         = require('socket.io')(server)
var http       = require('http')
var colors     = require('colors')
var config     = require("./config.json")

// set up app
var port = config.port
console.log(("Server is running at port " + port).magenta)
server.listen(port)

// global variable
var DEBUG = config.DEBUG

// including bingo.js
var bingo = require('./static/js/bingo_server.js')
var print = bingo.print 
bingo.init({ 
	'io':io,
	'config':config,
	'DEBUG':DEBUG
});

// middlewares
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
app.get('/spawn', function(req, res){
	res.sendFile(__dirname + '/pages/spawn_record.html', function(){
		res.end()
	})
})

// socket.io part
io.sockets.on('connection', function(socket){
	
	// initialize
	let player = new bingo.Player(socket)
	print("[Bingo] Someone connented.")
	socket.emit('refreshRoomInfo', { 'list':bingo.getSimpleRoomList() })

	// socket events
	socket.on('createRoomReq', function(data){
		if (!data.playerName || !data.roomName || !data.mode) return
		if (!player.name) player.name = escape(data.playerName.toString())
		data.roomName = escape(data.roomName.toString())
		createRoom(player, data)
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

	socket.on('downReq', function(num){
		if(player.rid) bingo.getRoomByRid(player.rid).downReq(player.id, num)
	})

	socket.on('AIConfigReq', function(data){
        if (!player.rid) return    
        let room = bingo.getRoomByRid(player.rid)
        if(room.playerList[0]!=player){
            print('Permission denied')
            return
        }
        room.AIConfig = data.config
        if(DEBUG) print("[Debug] Config changed:", room.AIConfig)
    })

	// for admin
	socket.on('removeRoom', function(rid, passwd){
		if (passwd != config.passwd) return
		bingo.getRoomByRid(rid).announce('message', {'message':'Admin removed the room.', 'msgId':bingo.randomString(8)})
		bingo.getRoomByRid(rid).removeRoom(bingo.roomList)
		io.emit('refreshRoomInfo', { 'list':bingo.getSimpleRoomList() })		
	})
    socket.on('set', function(params, value){
        if (params == 'DEBUG'){
            DEBUG = value
            bingo.init({
                'io':io,
                'config':config,
                'DEBUG':DEBUG
            })
        }
    })
	socket.on('check', function(params){
		if (params == 'roomList') print(bingo.getSimpleRoomList())
		if (params == 'id') print(player.id)
	})
})

// functions for Req process
function joinRoom(player, rid){
	player.joinRoom(bingo.getRoomByRid(rid))
	player.socket.emit('message', {'message':'Join room successfully.', 'msgId':bingo.randomString(8)})
}

function createRoom(player, data){
	if (data.mode != 'pvp' && data.mode != 'com') return
    if (data.mode == 'com'){
        let cnt = 0
        for (let i = 0; i < bingo.roomList.length; i++){
            if (bingo.roomList[i].mode == 'com') cnt++
        }
        if (cnt >= 2){
            player.socket.emit('message', {'message':'Server can hold only two AI.', 'msgId':bingo.randomString(8)})
            return
        }
    }
	let newRoom = new bingo.Room(data.roomName, data.mode)
    if(data.config) newRoom.AIConfig = data.config
	bingo.roomList.push(newRoom)
	player.joinRoom(newRoom);
	player.socket.emit('createRoomRes', {'status':'success','rid':newRoom.rid})
	io.emit('refreshRoomInfo', { 'list':bingo.getSimpleRoomList() })
	print("[Bingo] Player", player.name, "created room", data.roomName)
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
