var rdmString  = require('randomstring')
var spawn = require('child_process').spawn
var fs = require('fs')
var roomList = [];
var io,config,DEBUG;
var fileNum = 0, recordRoot = __dirname + "/../../Data/record/" + rdmString.generate(16) + '/';
fs.mkdir(recordRoot, function (err) {
	if(err) throw err;
});

function init(data){
	io = data.io
	config = data.config
	DEBUG = data.DEBUG
}

function Player(socket, name){
	this.socket = socket;
	this.name = name;
	this.id = socket.id
	this.rid = null;
	this.joinGame = function(room){
		if (!room.isJoinable()){
			this.socket.emit('joinGameRes',{'status':'failed', 'rid':room.rid})
			return
		}
		for (let i = 0; i < room.audList.length; ++i){
			if (room.audList[i] == this){
				room.audList.splice(i, 1)
				break
			}
		}
		room.playerList.push(this)
		this.socket.emit('joinGameRes', {'status':'success'})
		room.tryStart()
	}
	this.joinRoom = function(room){
		room.audList.push(this);
		this.leaveRoom();
		this.rid = room.rid;
		this.socket.emit('joinRoomRes', {'status':'success', 'rid':room.rid, 'joinable':room.isJoinable()})
        if (DEBUG){
            console.log("[Debug] Emit refreshstat command with")
        }
        this.socket.emit('refreshState', { stat: room.stat_1D, turn: room.turn })
	}
	this.leaveRoom = function(){
		let room = getRoomByRid(this.rid);
		if(!room) return
		for(let i = 0; i < room.playerList.length; ++i){
			if(room.playerList[i] == this){
				if(room.playing){
					let winnerId = i==0? 2:1
					if (DEBUG) console.log("[Debug] someone in the game left with mode", room.mode)
                    if (room.mode == 'com'){
                         room.gameOver({
                            'endWay':0,
                            'winnerId':winnerId,
                            'winnerName':'AI InforGo'
                        })
                    }
                    else {                        
				    	room.gameOver({
				    		'endWay': 0,
				    		'winnerId': winnerId,
				    		'winnerName': room.playerList[winnerId - 1].name
				    	})
                    }
				}
				room.playerList.splice(i, 1)
				if (room.isEmpty()) room.removeRoom()
			}
			// and declare gameOver
		}
		for(let i = 0; i < room.audList.length; i++ ){
			if(room.audList[i] == this){
				room.audList.splice(i, 1)
				if (room.isEmpty()) room.removeRoom()
			}
		}
	}
}

function Room(roomName, mode){
	this.name = roomName
	this.mode = mode  //( 'pvp', 'com' )
	this.rid = rdmString.generate(16);
	this.playerList = []
	this.audList = []
	this.record = ""
	this.stat_1D = []
	this.stat_3D = []
	this.playing = false
	this.turn = 0
	this.first = 1
    this.AIConfig = []
	for (let i = 0; i < 4; ++i){
		this.stat_3D[i] = []
		for (let j = 0; j < 4; ++j) this.stat_3D[i][j] = [0, 0, 0, 0]
	}
	this.initialize = function(){
		// initailize the stat array (turn the placable box into "3",the other to "0")
		for (let i = 0; i < 64; ++i){
		    if (i % 4) this.stat_1D[i] = 0
		    else this.stat_1D[i] = 3
		    this.stat_3D[i % 4][Math.floor(i / 4) % 4][Math.floor(i / 16)] = this.stat_1D[i]
		}
	}
	this.initialize(); // Important. If removed, clients might make box's material into null.
	this.isEmpty = function(){
		if (!this.playerList.length && !this.audList.length) return true
		return false
	}
	this.isJoinable = function(){
		if (this.mode == 'pvp'){
			if(this.playerList.length < 2) return true
			return false
		}
		if (this.mode == 'com'){
			if(this.playerList.length < 1) return true
			return false
		}
	}
	this.tryStart = function(){
		if (this.mode == 'pvp' && this.playerList.length == 2) this.pvpStart()
		if (this.mode == 'com' && this.playerList.length == 1) this.comStart()
		if (DEBUG) console.log("[Debug] Tried to start with \"mode {0}, player count {1}\".".format(this.mode, this.playerList.length))
	}
	this.pvpStart = function(){

		// setup variables and stats
		this.playing = true
		this.turn = 1
		this.initialize()

		// send messages to clients
		this.announce('restart')
		this.announce('refreshState', { stat: this.stat_1D, turn: this.turn })
		io.emit('refreshRoomInfo', { 'list':getSimpleRoomList() });
		this.playerList[0].socket.emit('playerAnnounce', 1)
		this.playerList[1].socket.emit('playerAnnounce', 2)
		console.log("[Bingo] Room", this.name, "game start!")
	}
	this.comStart = function(){
		
        if (DEBUG) console.log("[Debug] Called comstart function")
        let that = this

    	// set AI arguments
    	let option = ['-m', 'InforGo.main', 'run', '--directory=./Data/3_64_32_16/']
        option = option.concat(this.AIConfig)
        if (this.first == 1) option.push('--play_first=False')
        else option.push('--play_first=True')
        
        if (DEBUG){
        	console.log("[Debug] Start with option :", option)
        	this.announce('AIConfigRes', { 'config': option })
        }

        // set up agent

        this.agent = spawn('python', option ,{ cwd:__dirname+'/../../../'})
		this.agent.stdout.setEncoding('utf-8')
        this.agent.stdout.on('data', function(data){
            let agentDownId = checkRow(that.stat_1D, parseInt(data[0]), parseInt(data[2]))
            if (DEBUG) console.log("[Debug] AI down at {0}".format(agentDownId))
            that.agentDown(agentDownId) 
        })
        this.agent.stderr.on('data', (data) => {
            console.log("[Agent] Std error:",data)
        })
        this.agent.on('close', (code) => {
            console.log("[Agent] exit with code",code)
        })
        this.agent.on('error', (err) => {
            console.log("[Agent] Got output error:",err)
        })

        // setup variables, stat and agent
        this.playing = true
		this.turn = this.first
		this.initialize()
		this.announce('restart')
		this.announce('refreshState', { stat: this.stat_1D, turn: this.turn })
		io.emit('refreshRoomInfo', { 'list':getSimpleRoomList() });
		this.playerList[0].socket.emit('playerAnnounce', 1)
	}
	this.agentDown = function(num){

		// filter
		if (!this.playing) return
		if (this.stat_1D[num] != 3){
			console.log("[Bingo] WTFFFFFFFFFFFFFFFFFFFFFF, computer down at invalid place")
			return
		}
	
		// update stat_1D and record
		this.stat_1D[num] = this.turn
		this.stat_3D[num % 4][Math.floor(num / 4) % 4][Math.floor(num / 16)] = this.turn
		if ((num % 4) != 3) this.stat_1D[num + 1] = 3
		this.appendRecord(num)

		// check winner
		let winnerId = this.checkWinner(num)
		if (winnerId == 3) this.gameOver({'endWay': 3})
		if (winnerId == 0){
			this.turn = (this.turn == 1)? 2 : 1
			this.announce('refreshState', { stat: this.stat_1D, turn: this.turn, last: num })
		}
		else {
			// check if game is over
			if (winnerId == 2){
				this.gameOver({
					'endWay': winnerId,
					'winnerId': winnerId,
					'winnerName': 'AI InforGo'
				})
			}
			else {
				console.log("[Bingo] WTFFFFFFFFFFFFFFFFFF player win while AI down.")
			}
		}
	}
	this.downReq  = function(playerId, num) {

		// filter
		if (!this.playing) return
        if (DEBUG) console.log("[Debug] turn =", this.turn)
		if (!this.playerList[this.turn-1] || playerId != this.playerList[this.turn - 1].id) return
		if (this.stat_1D[num] != 3) return
		if (DEBUG) console.log("[Bingo]", "Player", this.playerList[this.turn - 1].name, "downed at", num)
	
		// update stat_1D and record
		this.stat_1D[num] = this.turn
		this.stat_3D[num % 4][Math.floor(num / 4) % 4][Math.floor(num / 16)] = this.turn
		if ((num % 4) != 3) this.stat_1D[num + 1] = 3
		this.appendRecord(num)

		// check winner
		let winnerId = this.checkWinner(num)
		if (winnerId == 3) this.gameOver({'endWay': 3})
		if (winnerId == 0){
			this.turn = (this.turn == 1)? 2 : 1
			this.announce('refreshState', { stat: this.stat_1D, turn: this.turn, last : num })
		    if (this.mode == 'com'){
                let writeString = (num % 4).toString() + ' ' + (Math.floor(num / 4) % 4).toString() + ' ' + (Math.floor(num / 16)).toString() + '\n'
                this.agent.stdin.write(writeString)
            }
        }
		else {
			// check if game is over
			this.gameOver({
				'endWay': winnerId,
				'winnerId': winnerId,
				'winnerName': this.playerList[winnerId-1].name
			})
		}
	}
	this.getCliById = function(Socket_id){
		for (let i = 0; i < this.audList.length; ++i) if (this.audList[i].id == Socket_id) return this.audList[i];
		for (let i = 0; i < this.playerList.length; ++i) if (this.playerList[i].id == Socket_id) return this.playerList[i];
	}
	this.gameOver = function(gameInfo){
		if (DEBUG) console.log("[Debug] called gameOver")

		// stop agent
		if (this.mode == 'com'){
			this.agent.stdin.write('-1 -1 -1')
			this.agent.stdin.end()
		}

		// clear table
		for (let i = 0; i < 64; ++i) if (this.stat_1D[i] == 3) this.stat_1D[i] = 0;

		//set variables
		this.playing = false;
		this.turn = 3;
		this.first = this.first == 1 ? 2 : 1

		// announce game info
		gameInfo.mode = this.mode
		this.announce('gameOver', gameInfo);
		this.announce('refreshState', { stat: this.stat_1D, turn: this.turn });

		// try start again
		let tmp = this;
		setTimeout(this.tryStart.bind(this), 10000);

		// write record
		if (gameInfo.endWay != 0) this.writeRecord();
		console.log("[Bingo] Game over with", gameInfo.endWay);
	}
	this.appendRecord = function(id){
		this.record += id % 4 + ' ';
		this.record += Math.floor(id / 4) % 4 + ' ';
		this.record += Math.floor(id / 16) + ' ';
		this.record += '\n';
	}
	this.writeRecord = function (){
	    this.record += "-1 -1 -1\n"
	    let recordPath = recordRoot + fileNum.toString();
        fs.writeFile(recordPath, this.record, function (err) {
	    	if(err) throw err;
	    })
        this.record = ""
        fileNum++;
    }
	this.checkWinner = function(id){
		// check if the game is over.
		let x = id % 4;
		let y = Math.floor(id / 4) % 4;
		let z = Math.floor(id / 16);
		let value = 1;
		let sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		for (let i = 0; i < 4; ++i, value *= 2){
			let point = [
				this.stat_3D[x][y][i], this.stat_3D[x][i][z], this.stat_3D[i][y][z], // 1D
				this.stat_3D[x][i][i], this.stat_3D[i][y][i], this.stat_3D[i][i][z], // 2D
				this.stat_3D[x][i][3-i],  this.stat_3D[i][y][3-i], this.stat_3D[i][3-i][z], //2D_inverse
				this.stat_3D[i][i][i], this.stat_3D[i][i][3-i], this.stat_3D[3-i][i][i], // 3D
				this.stat_3D[3-i][i][3-i]
			];
			for (let line = 0; line < point.length; ++line){
				if(point[line] == 1) sum[line] += value;
				if(point[line] == 2) sum[line] -= value;
			}
		}

		// return 1 if there is a line with value "15", ortherwise return 2 if got value"-15". if not, return 0.
		for (let i = 0; i < sum.length; ++i){
			if(sum[i] == 15) return 1;
			if(sum[i] ==-15) return 2;
		}
		// return 0 if the game could keep going on.
		for (let i = 0; i < 64; ++i){
			if(this.stat_1D[i] == 3)return 0;
		}
		// return 3 if there is no empty place.
		return 3;
	}
	this.removeRoom = function(){
		io.emit('message', {'message':'Room removed.', 'msgId':randomString(8)})
		if (DEBUG) console.log('[Debug] Room', this.rid, 'removed.');
		for(let i = 0; i < roomList.length; ++i){
			if (roomList[i].rid == this.rid){
				roomList.splice(i, 1);
				break;
			}
		}
		io.emit('refreshRoomInfo', { 'list':getSimpleRoomList() });
	}
	this.announce = function(event, data){
		for(let i = 0; i < this.playerList.length; ++i){
			if (data) this.playerList[i].socket.emit(event, data);
			else this.playerList[i].socket.emit(event);
		}
		for(let i = 0; i < this.audList.length; ++i){
			if (data) this.audList[i].socket.emit(event ,data);
			else this.audList[i].socket.emit(event);
		}
	}
}

function randomString(length){
	return rdmString.generate(length);
}

function playerDisconnect(player){
	player.leaveRoom();
	console.log("[Bingo] someone disconnect")
}

function getRoomByRid(rid){
	if (!rid) return
	for (let i = 0; i < roomList.length; ++i){
		if (roomList[i].rid == rid) return roomList[i]
	}
	throw "Room not found"
}

function getSimpleRoomList(){
	let simpleList = []
	for(let i = 0; i < roomList.length; ++i){
		simpleList.push({
			'rid':     roomList[i].rid,
			'name':    roomList[i].name,
			'mode':    roomList[i].mode,
			'playing': roomList[i].playing,
			'joinable': roomList[i].isJoinable()
		})
	}
	return simpleList
}

function checkRow(stat,x, y){
    let rowId = x*4 + y*16
    for (let i = 0; i<4;i++){
        if (stat[rowId + i] == 3) return rowId + i
    }
    console.log("[Bingo] WTFFFFFFFFFFFFF Full row checked!!")
    return
}

// string format function
String.prototype.format = function(){
    let s = this, i = arguments.length;
    while(i--)s = s.replace(new RegExp('\\{'+i+'\\}', 'gm'), arguments[i]);
    return s;
};

module.exports = {
	init,
	Room,
	Player,
	randomString,
	roomList,
	playerDisconnect,
	getRoomByRid,
	//getRoomByPlayer,
	getSimpleRoomList,
}
