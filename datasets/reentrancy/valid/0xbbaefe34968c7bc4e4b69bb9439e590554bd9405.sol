pragma solidity ^0.4.24;

 

 
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}

 

 
library SafeMath {

   
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
     
     
     
    if (a == 0) {
      return 0;
    }

    c = a * b;
    assert(c / a == b);
    return c;
  }

   
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
     
     
     
    return a / b;
  }

   
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

   
  function add(uint256 a, uint256 b) internal pure returns (uint256 c) {
    c = a + b;
    assert(c >= a);
    return c;
  }
}

 

 
contract BasicToken is ERC20Basic {
  using SafeMath for uint256;

  mapping(address => uint256) balances;

  uint256 totalSupply_;

   
  function totalSupply() public view returns (uint256) {
    return totalSupply_;
  }

   
  function transfer(address _to, uint256 _value) public returns (bool) {
    require(_to != address(0));
    require(_value <= balances[msg.sender]);

    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    emit Transfer(msg.sender, _to, _value);
    return true;
  }

   
  function balanceOf(address _owner) public view returns (uint256) {
    return balances[_owner];
  }

}

 

 
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender)
    public view returns (uint256);

  function transferFrom(address from, address to, uint256 value)
    public returns (bool);

  function approve(address spender, uint256 value) public returns (bool);
  event Approval(
    address indexed owner,
    address indexed spender,
    uint256 value
  );
}

 

 
contract StandardToken is ERC20, BasicToken {

  mapping (address => mapping (address => uint256)) internal allowed;


   
  function transferFrom(
    address _from,
    address _to,
    uint256 _value
  )
    public
    returns (bool)
  {
    require(_to != address(0));
    require(_value <= balances[_from]);
    require(_value <= allowed[_from][msg.sender]);

    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    emit Transfer(_from, _to, _value);
    return true;
  }

   
  function approve(address _spender, uint256 _value) public returns (bool) {
    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
  }

   
  function allowance(
    address _owner,
    address _spender
   )
    public
    view
    returns (uint256)
  {
    return allowed[_owner][_spender];
  }

   
  function increaseApproval(
    address _spender,
    uint256 _addedValue
  )
    public
    returns (bool)
  {
    allowed[msg.sender][_spender] = (
      allowed[msg.sender][_spender].add(_addedValue));
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

   
  function decreaseApproval(
    address _spender,
    uint256 _subtractedValue
  )
    public
    returns (bool)
  {
    uint256 oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
      allowed[msg.sender][_spender] = 0;
    } else {
      allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

}

 

 
contract Ownable {
  address public owner;


  event OwnershipRenounced(address indexed previousOwner);
  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );


   
  constructor() public {
    owner = msg.sender;
  }

   
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }

   
  function renounceOwnership() public onlyOwner {
    emit OwnershipRenounced(owner);
    owner = address(0);
  }

   
  function transferOwnership(address _newOwner) public onlyOwner {
    _transferOwnership(_newOwner);
  }

   
  function _transferOwnership(address _newOwner) internal {
    require(_newOwner != address(0));
    emit OwnershipTransferred(owner, _newOwner);
    owner = _newOwner;
  }
}

 

 
contract Pausable is Ownable {
  event Pause();
  event Unpause();

  bool public paused = false;


   
  modifier whenNotPaused() {
    require(!paused);
    _;
  }

   
  modifier whenPaused() {
    require(paused);
    _;
  }

   
  function pause() onlyOwner whenNotPaused public {
    paused = true;
    emit Pause();
  }

   
  function unpause() onlyOwner whenPaused public {
    paused = false;
    emit Unpause();
  }
}

 

contract MarketDataStorage is Ownable {
     
    address[] supportedTokens;
    mapping (address => bool) public supportedTokensMapping;  
    mapping (address => uint[]) public currentTokenMarketData;  
    mapping (bytes32 => bool) internal validIds;  
    address dataUpdater;  

     
    modifier updaterOnly() {
        require(
            msg.sender == dataUpdater,
            "updater not allowed"
        );
        _;
    }

    modifier supportedTokenOnly(address token_address) {
        require(
            isTokenSupported(token_address),
            "Can't update a non supported token"
        );
        _;
    }

    constructor (address[] _supportedTokens, address _dataUpdater) Ownable() public {
        dataUpdater = _dataUpdater;

         
        for (uint i=0; i<_supportedTokens.length; i++) {
            addSupportedToken(_supportedTokens[i]);
        }
    }

    function numberOfSupportedTokens() view public returns (uint) {
        return supportedTokens.length;
    }

    function getSupportedTokenByIndex(uint idx) view public returns (address token_address, bool supported_status) {
        address token = supportedTokens[idx];
        return (token, supportedTokensMapping[token]);
    }

    function getMarketDataByTokenIdx(uint idx) view public returns (address token_address, uint volume, uint depth, uint marketcap) {
        (address token, bool status) = getSupportedTokenByIndex(idx);

        (uint _volume, uint _depth, uint _marketcap) = getMarketData(token);

        return (token, _volume, _depth, _marketcap);
    }

    function getMarketData(address token_address) view public returns (uint volume, uint depth, uint marketcap) {
         
        if (!supportedTokensMapping[token_address]) {
            return (0,0,0);
        }

        uint[] memory data = currentTokenMarketData[token_address];
        return (data[0], data[1], data[2]);
    }

    function addSupportedToken(address token_address) public onlyOwner {
        require(
            isTokenSupported(token_address) == false,
            "Token already added"
        );

        supportedTokens.push(token_address);
        supportedTokensMapping[token_address] = true;

        currentTokenMarketData[token_address] = [0,0,0];  
    }

    function isTokenSupported(address token_address) view public returns (bool) {
        return supportedTokensMapping[token_address];
    }

     
    function updateMarketData(address token_address,
        uint volume,
        uint depth,
        uint marketcap)
    external
    updaterOnly
    supportedTokenOnly(token_address) {
        currentTokenMarketData[token_address] = [volume,depth,marketcap];
    }
}

 

contract WarOfTokens is Pausable {
    using SafeMath for uint256;

    struct AttackInfo {
        address attacker;
        address attackee;
        uint attackerScore;
        uint attackeeScore;
        bytes32 attackId;
        bool completed;
        uint hodlSpellBlockNumber;
        mapping (address => uint256) attackerWinnings;
        mapping (address => uint256) attackeeWinnings;
    }

     
    event Deposit(address token, address user, uint amount, uint balance);
    event Withdraw(address token, address user, uint amount, uint balance);
    event UserActiveStatusChanged(address user, bool isActive);
    event Attack (
        address indexed attacker,
        address indexed attackee,
        bytes32 attackId,
        uint attackPrizePercent,
        uint base,
        uint hodlSpellBlockNumber
    );
    event AttackCompleted (
        bytes32 indexed attackId,
        address indexed winner,
        uint attackeeActualScore
    );

     
     
    mapping (address => mapping (address => uint256)) public tokens;
    mapping (address => bool) public activeUsers;
    address public cdtTokenAddress;
    uint256 public minCDTToParticipate;
    MarketDataStorage public marketDataOracle;
    uint public maxAttackPrizePercent;  
    uint attackPricePrecentBase = 1000;  
    uint public maxOpenAttacks = 5;
    mapping (bytes32 => AttackInfo) public attackIdToInfo;
    mapping (address => mapping(address => bytes32)) public userToUserToAttackId;
    mapping (address => uint) public cntUserAttacks;  


     
    modifier activeUserOnly(address user) {
        require(
            isActiveUser(user),
            "User not active"
        );
        _;
    }

    constructor(address _cdtTokenAddress,
        uint256 _minCDTToParticipate,
        address _marketDataOracleAddress,
        uint _maxAttackPrizeRatio)
    Pausable()
    public {
        cdtTokenAddress = _cdtTokenAddress;
        minCDTToParticipate = _minCDTToParticipate;
        marketDataOracle = MarketDataStorage(_marketDataOracleAddress);
        setMaxAttackPrizePercent(_maxAttackPrizeRatio);
    }

     
    function() public {
        revert("Please do not send ETH without calling the deposit function. We will not do it automatically to validate your intent");
    }

     
    function isActiveUser(address user) view public returns (bool) {
        return activeUsers[user];
    }

     
     
     
     
     

     
     
    function deposit() payable external whenNotPaused {
        tokens[0][msg.sender] = tokens[0][msg.sender].add(msg.value);
        emit Deposit(0, msg.sender, msg.value, tokens[0][msg.sender]);

        _validateUserActive(msg.sender);
    }

     
    function depositToken(address token, uint amount) external whenNotPaused {
         
        require(
            token!=0,
            "unrecognized token"
        );
        assert(StandardToken(token).transferFrom(msg.sender, this, amount));
        tokens[token][msg.sender] =  tokens[token][msg.sender].add(amount);
        emit Deposit(token, msg.sender, amount, tokens[token][msg.sender]);

        _validateUserActive(msg.sender);
    }

    function withdraw(uint amount) external {
        tokens[0][msg.sender] = tokens[0][msg.sender].sub(amount);
        assert(msg.sender.call.value(amount)());
        emit Withdraw(0, msg.sender, amount, tokens[0][msg.sender]);

        _validateUserActive(msg.sender);
    }

    function withdrawToken(address token, uint amount) external {
        require(
            token!=0,
            "unrecognized token"
        );
        tokens[token][msg.sender] = tokens[token][msg.sender].sub(amount);
        assert(StandardToken(token).transfer(msg.sender, amount));
        emit Withdraw(token, msg.sender, amount, tokens[token][msg.sender]);

        _validateUserActive(msg.sender);
    }

    function balanceOf(address token, address user) view public returns (uint) {
        return tokens[token][user];
    }

     
     
     
     
     
    function setMaxAttackPrizePercent(uint newAttackPrize) onlyOwner public {
        require(
            newAttackPrize < 5,
            "max prize is 5 percent of funds"
        );
        maxAttackPrizePercent = newAttackPrize;
    }

    function setMaxOpenAttacks(uint newValue) onlyOwner public {
        maxOpenAttacks = newValue;
    }

    function openAttacksCount(address user) view public returns (uint) {
        return cntUserAttacks[user];
    }

    function isTokenSupported(address token_address) view public returns (bool) {
        return marketDataOracle.isTokenSupported(token_address);
    }

    function getUserScore(address user)
    view
    public
    whenNotPaused
    returns (uint) {
        uint cnt_supported_tokens = marketDataOracle.numberOfSupportedTokens();
        uint aggregated_score = 0;
        for (uint i=0; i<cnt_supported_tokens; i++) {
            (address token_address, uint volume, uint depth, uint marketcap) = marketDataOracle.getMarketDataByTokenIdx(i);
            uint256 user_balance = balanceOf(token_address, user);

            aggregated_score = aggregated_score + _calculateScore(user_balance, volume, depth, marketcap);
        }

        return aggregated_score;
    }

    function _calculateScore(uint256 balance, uint volume, uint depth, uint marketcap) pure internal returns (uint) {
        return balance * volume * depth * marketcap;
    }

    function attack(address attackee)
    external
    activeUserOnly(msg.sender)
    activeUserOnly(attackee)
    {
        require(
            msg.sender != attackee,
            "Can't attack yourself"
        );
        require(
            userToUserToAttackId[msg.sender][attackee] == 0,
            "Cannot attack while pending attack exists, please complete attack"
        );
        require(
            openAttacksCount(msg.sender) < maxOpenAttacks,
            "Too many open attacks for attacker"
        );
        require(
            openAttacksCount(attackee) < maxOpenAttacks,
            "Too many open attacks for attackee"
        );

        (uint attackPrizePercent, uint attackerScore, uint attackeeScore) = attackPrizeRatio(attackee);

        AttackInfo memory attackInfo = AttackInfo(
            msg.sender,
            attackee,
            attackerScore,
            attackeeScore,
            sha256(abi.encodePacked(msg.sender, attackee, block.blockhash(block.number-1))),  
            false,
            block.number  
        );
        _registerAttack(attackInfo);

        _calculateWinnings(attackIdToInfo[attackInfo.attackId], attackPrizePercent);

        emit Attack(
            attackInfo.attacker,
            attackInfo.attackee,
            attackInfo.attackId,
            attackPrizePercent,
            attackPricePrecentBase,
            attackInfo.hodlSpellBlockNumber
        );
    }

     
    function attackPrizeRatio(address attackee)
    view
    public
    returns (uint attackPrizePercent, uint attackerScore, uint attackeeScore) {
        uint _attackerScore = getUserScore(msg.sender);
        require(
            _attackerScore > 0,
            "attacker score is 0"
        );
        uint _attackeeScore = getUserScore(attackee);
        require(
            _attackeeScore > 0,
            "attackee score is 0"
        );

        uint howCloseAreThey = _attackeeScore.mul(attackPricePrecentBase).div(_attackerScore);

        return (howCloseAreThey, _attackerScore, _attackeeScore);
    }

    function attackerPrizeByToken(bytes32 attackId, address token_address) view public returns (uint256) {
        return attackIdToInfo[attackId].attackerWinnings[token_address];
    }

    function attackeePrizeByToken(bytes32 attackId, address token_address) view public returns (uint256) {
        return attackIdToInfo[attackId].attackeeWinnings[token_address];
    }

     
    function completeAttack(bytes32 attackId) public {
        AttackInfo storage attackInfo = attackIdToInfo[attackId];

        (address winner, uint attackeeActualScore) = getWinner(attackId);

         
        uint cnt_supported_tokens = marketDataOracle.numberOfSupportedTokens();
        for (uint i=0; i<cnt_supported_tokens; i++) {
            (address token_address, bool status) = marketDataOracle.getSupportedTokenByIndex(i);

            if (attackInfo.attacker == winner) {
                uint winnings = attackInfo.attackerWinnings[token_address];

                if (winnings > 0) {
                    tokens[token_address][attackInfo.attackee] = tokens[token_address][attackInfo.attackee].sub(winnings);
                    tokens[token_address][attackInfo.attacker] = tokens[token_address][attackInfo.attacker].add(winnings);
                }
            }
            else {
                uint loosings = attackInfo.attackeeWinnings[token_address];

                if (loosings > 0) {
                    tokens[token_address][attackInfo.attacker] = tokens[token_address][attackInfo.attacker].sub(loosings);
                    tokens[token_address][attackInfo.attackee] = tokens[token_address][attackInfo.attackee].add(loosings);
                }
            }
        }

         
        _unregisterAttack(attackId);

        emit AttackCompleted(
            attackId,
            winner,
            attackeeActualScore
        );
    }

    function getWinner(bytes32 attackId) public view returns(address winner, uint attackeeActualScore) {
        require(
            block.number >= attackInfo.hodlSpellBlockNumber,
            "attack can not be completed at this block, please wait"
        );

        AttackInfo storage attackInfo = attackIdToInfo[attackId];

         
         
         
         
        if (block.number - attackInfo.hodlSpellBlockNumber >= 256) {
            return (attackInfo.attackee, attackInfo.attackeeScore);
        }

        bytes32 blockHash = block.blockhash(attackInfo.hodlSpellBlockNumber);
        return _calculateWinnerBasedOnEntropy(attackInfo, blockHash);
    }

     
     
     
     
     

     
    function _validateUserActive(address user) private {
         
        uint256 cdt_balance = balanceOf(cdtTokenAddress, user);

        bool new_active_state = cdt_balance >= minCDTToParticipate;
        bool current_active_state = activeUsers[user];  

        if (current_active_state != new_active_state) {  
            emit UserActiveStatusChanged(user, new_active_state);
        }

        activeUsers[user] = new_active_state;
    }

    function _registerAttack(AttackInfo attackInfo) internal {
        userToUserToAttackId[attackInfo.attacker][attackInfo.attackee] = attackInfo.attackId;
        userToUserToAttackId[attackInfo.attackee][attackInfo.attacker] = attackInfo.attackId;

        attackIdToInfo[attackInfo.attackId] = attackInfo;

         
        cntUserAttacks[attackInfo.attacker] = cntUserAttacks[attackInfo.attacker].add(1);
        cntUserAttacks[attackInfo.attackee] = cntUserAttacks[attackInfo.attackee].add(1);
    }

    function _unregisterAttack(bytes32 attackId) internal {
        AttackInfo storage attackInfo = attackIdToInfo[attackId];

        cntUserAttacks[attackInfo.attacker] = cntUserAttacks[attackInfo.attacker].sub(1);
        cntUserAttacks[attackInfo.attackee] = cntUserAttacks[attackInfo.attackee].sub(1);

        delete userToUserToAttackId[attackInfo.attacker][attackInfo.attackee];
        delete userToUserToAttackId[attackInfo.attackee][attackInfo.attacker];

        delete attackIdToInfo[attackId];
    }

     
    function _calculateWinnings(AttackInfo storage attackInfo, uint attackPrizePercent) internal {
         
        uint cnt_supported_tokens = marketDataOracle.numberOfSupportedTokens();

        uint actualPrizeRation = attackPrizePercent
        .mul(maxAttackPrizePercent);


        for (uint i=0; i<cnt_supported_tokens; i++) {
            (address token_address, bool status) = marketDataOracle.getSupportedTokenByIndex(i);

            if (status) {
                 
                uint256 _b1 = balanceOf(token_address, attackInfo.attacker);
                if (_b1 > 0) {
                    uint256 _w1 = _b1.mul(actualPrizeRation).div(attackPricePrecentBase * 100);  
                    attackInfo.attackeeWinnings[token_address] = _w1;
                }

                 
                uint256 _b2 = balanceOf(token_address, attackInfo.attackee);
                if (_b2 > 0) {
                    uint256 _w2 = _b2.mul(actualPrizeRation).div(attackPricePrecentBase * 100);  
                    attackInfo.attackerWinnings[token_address] = _w2;
                }
            }
        }
    }

     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
    function _calculateWinnerBasedOnEntropy(AttackInfo storage attackInfo, bytes32 entropy) view internal returns(address, uint) {
        uint attackeeActualScore = attackInfo.attackeeScore;
        uint modul = _absSubtraction(attackInfo.attackerScore, attackInfo.attackeeScore);
        modul = modul.mul(2);  
        uint hodlSpell = uint(entropy) % modul;
        uint direction = uint(entropy) % 10;
        uint directionThreshold = 1;

         
         
        if (attackInfo.attackerScore < attackInfo.attackeeScore) {
            directionThreshold = 8;
        }

         
        if (direction > directionThreshold) {
            attackeeActualScore = attackeeActualScore.add(hodlSpell);
        }
        else {
            attackeeActualScore = _safeSubtract(attackeeActualScore, hodlSpell);
        }
        if (attackInfo.attackerScore > attackeeActualScore) { return (attackInfo.attacker, attackeeActualScore); }
        else { return (attackInfo.attackee, attackeeActualScore); }
    }

     
     
     
    function _absSubtraction(uint a, uint b) pure internal returns (uint) {
        if (b>a) {
            return b-a;
        }

        return a-b;
    }

     
     
    function _safeSubtract(uint a, uint b) pure internal returns (uint) {
        if (b > a) {
            return 0;
        }

        return a-b;
    }
}