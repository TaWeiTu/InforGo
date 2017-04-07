var rdmString  = require('randomstring');
var spawn = require('child_process').spawn
function Player(socket, name){
	this.socket = socket;
	this.name = name;
	this.joinGame = function(room){
		room.playerList.push(this);
		for(let i=0;i<room.audList.length;i++){
			if(room.audList[i] == this){
				room.audList.splice(i,1);
			}
		}
	}
	this.joinRoom = function(room){
		room.audList.push(this);
	}
	this.leaveRoom = function(room){
		for(let i=0;i<room.playerList.length;i++){
			if(room.playerList[i] == this){
				room.playerList.splice(i,1);
			}
			// and declare gameOver
		}
		for(let i=0;i<room.audList.length;i++ ){
			if(room.audList[i] == this){
				room.audList.splice(i,1);
			}
		}
	}
}
function Room(name, mode){
	this.name = name;
	this.rid = rdmString.generate(16);
	this.playerList = [];
	this.audList = [];
	this.record = "";
	this.mode = mode; //( 'pvp', 'com' )
	this.stat_1D = [];
	this.stat_3D = [];
	this.playing = false;
	for(let i=0;i<4;i++){
		this.stat_3D[i] = [];
		for(let j=0;j<4;j++)this.stat_3D[i][j] = [0, 0, 0, 0];
	}
	this.initialize = function(){
		// initailize the stat array (turn the placable box into "3",the other to "0")
		for(let i=0;i<64;i++){
		    if(i%4)this.stat_1D[i] = 0;
		    else this.stat_1D[i] = 3;
		    this.stat_3D[i%4][Math.floor(i/4)%4][Math.floor(i/16)] = this.stat_1D[i];
		}
	}
	this.initialize(); // Important. If removed, clients might make box's material into null.
	this.tryStart = function(){
		if(this.playing)return;
		let py = spawn('python', ['../py/bingo.py', 'n_hidden_layer', '3', 'n_node_hidden', '32', '16', '8']);
		//renew player list
		this.playerList[0] = waitingList.shift(1);
		this.playerList[1] = waitingList.shift(1);
		//setup variables and stats
		this.record = "";
		this.playing = true;
		this.player = 1;
		initialize();
		//send messages to clients
		io.emit('restart');
		io.emit('refreshState', this.stat_1D, this.player);
		io.to(this.playerList[0]["id"]).emit('youare', 1);
		io.to(this.playerList[1]["id"]).emit('youare', 2);
		console.log(colors.title("[Bingo] ") + "Game start!");
	}
	this.joinGame = function(Socket_id){
	}
	this.downReq  = function(player_id, num) {
		// filter
		if(!this.playing)return;
		if(player_id != this.playerList[player-1]["id"])return;
		if(this.stat_1D[num] != 3)return;
		// update stat_1D and record
		this.stat_1D[num] = this.player;
		if((num%4) != 3)this.stat_1D[num+1] = 3;
		this.appendRecord(num);

		// check winner
		let winnerId = checkWinner(num);
		if(winnerId == 3)gameOver({'endWay': 3});
		if(winnerId == 0){
			this.player = (this.player == 1)?2:1;
			io.emit('refreshState', this.stat_1D, this.player);
		}
		else{
			// check if game is over
			gameOver({
				'endWay': winnerId,
				'winnerId': winnerId,
				'winnerName': this.playerList[winnerId-1].name
			});
			waitingList.push(playerList[1]);
			waitingList.push(playerList[0]);
		}
	}
	this.getCliById = function(Socket_id){
		for(let i=0;i<this.audList.length; i++)if(this.audList[i].id == Socket_id)return this.audList[i];
		for(let i=0;i<this.playerList.length; i++)if(this.playerList[i].id == Socket_id)return this.playerList[i];
	}
	this.refresh = function(){
	}
	this.gameOver = function(gameInfo){
		for(let i=0;i<64;i++)if(this.stat_1D[i] == 3)this.stat_1D[i]=0;
		io.emit('gameOver', gameInfo);
		io.emit('refreshState', this.stat_1D, 3);
		this.playing = false;
		this.player = 3;
		setTimeout(tryStart, 10000);
		if(gameInfo.endWay != 0)writeRecord();
		console.log(colors.title("[Bingo] ")+colors.info("Game over."));
	}
	this.appendRecord = function(id){
		this.record += id%4 + ' ';
		this.record += Math.floor(id/16) + ' ';
		this.record += Math.floor(id/4)%4 + ' ';
		this.record += '\n';
	}
	this.checkWinner = function(id){
		// check if the game is over.
		let x = id%4;
		let y = Math.floor(id/4)%4;
		let z = Math.floor(id/16);
		let value = 1;
		let sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		for(let i=0;i<4;i++,value*=2){
			let point = [
				this.stat_3D[x][y][i], this.stat_3D[x][i][z], this.stat_3D[i][y][z], // 1D
				this.stat_3D[x][i][i], this.stat_3D[i][y][i], this.stat_3D[i][i][z], // 2D
				this.stat_3D[x][i][3-i],  this.stat_3D[i][y][3-i], this.stat_3D[i][3-i][z], //2D_inverse
				this.stat_3D[i][i][i], this.stat_3D[i][i][3-i], this.stat_3D[3-i][i][i], // 3D
				this.stat_3D[3-i][i][3-i]
			];
			for(let line=0;line<point.length;line++){
				if(point[line] == 1)sum[line] += value;
				if(point[line] == 2)sum[line] -= value;
			}
		}

		// return 1 if there is a line with value "15", ortherwise return 2 if got value"-15". if not, return 0.
		for(let i=0;i<sum.length;i++){
			if(sum[i] == 15)return 1;
			if(sum[i] ==-15)return 2;
		}
		// return 0 if the game could keep going on.
		for(let i=0;i<64;i++){
			if(this.stat_1D[i] == 3)return 0;
		}
		// return 3 if there is no empty place.
		return 3;
	}
	this.removeRoom = function(roomList, io){
		io.emit('message',{'message':'Room removed.','id':this.rid})
		console.log('Room',this.rid,'is getting removed.');
		for(let i=0;i<roomList.length;i++){
			if (roomList[i] == this){
				roomList.splice(i, 1);
				break;
			}
		}
	}
}
function randomString(length){
	return rdmString.generate(length);
}

module.exports = {
	Room,
	Player,
	randomString
}
