pragma solidity ^0.4.24;

 

 
library SafeMath {

   
  function mul(uint256 _a, uint256 _b) internal pure returns (uint256 c) {
     
     
     
    if (_a == 0) {
      return 0;
    }

    c = _a * _b;
    assert(c / _a == _b);
    return c;
  }

   
  function div(uint256 _a, uint256 _b) internal pure returns (uint256) {
     
     
     
    return _a / _b;
  }

   
  function sub(uint256 _a, uint256 _b) internal pure returns (uint256) {
    assert(_b <= _a);
    return _a - _b;
  }

   
  function add(uint256 _a, uint256 _b) internal pure returns (uint256 c) {
    c = _a + _b;
    assert(c >= _a);
    return c;
  }
}

 

 
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address _who) public view returns (uint256);
  function transfer(address _to, uint256 _value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}

 

 
contract ERC20 is ERC20Basic {
  function allowance(address _owner, address _spender)
    public view returns (uint256);

  function transferFrom(address _from, address _to, uint256 _value)
    public returns (bool);

  function approve(address _spender, uint256 _value) public returns (bool);
  event Approval(
    address indexed owner,
    address indexed spender,
    uint256 value
  );
}

 

contract IBasicMultiToken is ERC20 {
    event Bundle(address indexed who, address indexed beneficiary, uint256 value);
    event Unbundle(address indexed who, address indexed beneficiary, uint256 value);

    ERC20[] public tokens;

    function tokensCount() public view returns(uint256);

    function bundleFirstTokens(address _beneficiary, uint256 _amount, uint256[] _tokenAmounts) public;
    function bundle(address _beneficiary, uint256 _amount) public;

    function unbundle(address _beneficiary, uint256 _value) public;
    function unbundleSome(address _beneficiary, uint256 _value, ERC20[] _tokens) public;

    function disableBundling() public;
    function enableBundling() public;
}

 

contract IMultiToken is IBasicMultiToken {
    event Update();
    event Change(address indexed _fromToken, address indexed _toToken, address indexed _changer, uint256 _amount, uint256 _return);

    mapping(address => uint256) public weights;

    function getReturn(address _fromToken, address _toToken, uint256 _amount) public view returns (uint256 returnAmount);
    function change(address _fromToken, address _toToken, uint256 _amount, uint256 _minReturn) public returns (uint256 returnAmount);

    function disableChanges() public;
}

 

library CheckedERC20 {
    using SafeMath for uint;

    function isContract(address addr) internal view returns(bool result) {
         
        assembly {
            result := gt(extcodesize(addr), 0)
        }
    }

    function handleReturnBool() internal pure returns(bool result) {
         
        assembly {
            switch returndatasize()
            case 0 {  
                result := 1
            }
            case 32 {  
                returndatacopy(0, 0, 32)
                result := mload(0)
            }
            default {  
                revert(0, 0)
            }
        }
    }

    function handleReturnBytes32() internal pure returns(bytes32 result) {
         
        assembly {
            if eq(returndatasize(), 32) {  
                returndatacopy(0, 0, 32)
                result := mload(0)
            }
            if gt(returndatasize(), 32) {  
                returndatacopy(0, 64, 32)
                result := mload(0)
            }
            if lt(returndatasize(), 32) {  
                revert(0, 0)
            }
        }
    }

    function asmTransfer(address _token, address _to, uint256 _value) internal returns(bool) {
        require(isContract(_token));
         
        require(_token.call(bytes4(keccak256("transfer(address,uint256)")), _to, _value));
        return handleReturnBool();
    }

    function asmTransferFrom(address _token, address _from, address _to, uint256 _value) internal returns(bool) {
        require(isContract(_token));
         
        require(_token.call(bytes4(keccak256("transferFrom(address,address,uint256)")), _from, _to, _value));
        return handleReturnBool();
    }

    function asmApprove(address _token, address _spender, uint256 _value) internal returns(bool) {
        require(isContract(_token));
         
        require(_token.call(bytes4(keccak256("approve(address,uint256)")), _spender, _value));
        return handleReturnBool();
    }

     

    function checkedTransfer(ERC20 _token, address _to, uint256 _value) internal {
        if (_value > 0) {
            uint256 balance = _token.balanceOf(this);
            asmTransfer(_token, _to, _value);
            require(_token.balanceOf(this) == balance.sub(_value), "checkedTransfer: Final balance didn't match");
        }
    }

    function checkedTransferFrom(ERC20 _token, address _from, address _to, uint256 _value) internal {
        if (_value > 0) {
            uint256 toBalance = _token.balanceOf(_to);
            asmTransferFrom(_token, _from, _to, _value);
            require(_token.balanceOf(_to) == toBalance.add(_value), "checkedTransfer: Final balance didn't match");
        }
    }

     

    function asmName(address _token) internal view returns(bytes32) {
        require(isContract(_token));
         
        require(_token.call(bytes4(keccak256("name()"))));
        return handleReturnBytes32();
    }

    function asmSymbol(address _token) internal view returns(bytes32) {
        require(isContract(_token));
         
        require(_token.call(bytes4(keccak256("symbol()"))));
        return handleReturnBytes32();
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

 

 
library SafeERC20 {
  function safeTransfer(
    ERC20Basic _token,
    address _to,
    uint256 _value
  )
    internal
  {
    require(_token.transfer(_to, _value));
  }

  function safeTransferFrom(
    ERC20 _token,
    address _from,
    address _to,
    uint256 _value
  )
    internal
  {
    require(_token.transferFrom(_from, _to, _value));
  }

  function safeApprove(
    ERC20 _token,
    address _spender,
    uint256 _value
  )
    internal
  {
    require(_token.approve(_spender, _value));
  }
}

 

 
contract CanReclaimToken is Ownable {
  using SafeERC20 for ERC20Basic;

   
  function reclaimToken(ERC20Basic _token) external onlyOwner {
    uint256 balance = _token.balanceOf(this);
    _token.safeTransfer(owner, balance);
  }

}

 

contract IEtherToken is ERC20 {
    function deposit() public payable;
    function withdraw(uint256 _amount) public;
}


contract IBancorNetwork {
    function convert(
        address[] _path,
        uint256 _amount,
        uint256 _minReturn
    ) 
        public
        payable
        returns(uint256);

    function claimAndConvert(
        address[] _path,
        uint256 _amount,
        uint256 _minReturn
    ) 
        public
        payable
        returns(uint256);
}


contract IKyberNetworkProxy {
    function trade(
        address src,
        uint srcAmount,
        address dest,
        address destAddress,
        uint maxDestAmount,
        uint minConversionRate,
        address walletId
    )
        public
        payable
        returns(uint);
}


contract MultiChanger is CanReclaimToken {
    using SafeMath for uint256;
    using CheckedERC20 for ERC20;

     
     
     
    function externalCall(address destination, uint value, bytes data, uint dataOffset, uint dataLength) internal returns (bool result) {
         
        assembly {
            let x := mload(0x40)    
            let d := add(data, 32)  
            result := call(
                sub(gas, 34710),    
                                    
                                    
                destination,
                value,
                add(d, dataOffset),
                dataLength,         
                x,
                0                   
            )
        }
    }

    function change(bytes _callDatas, uint[] _starts) public payable {  
        for (uint i = 0; i < _starts.length - 1; i++) {
            require(externalCall(this, 0, _callDatas, _starts[i], _starts[i + 1] - _starts[i]));
        }
    }

    function sendEthValue(address _target, bytes _data, uint256 _value) external {
         
        require(_target.call.value(_value)(_data));
    }

    function sendEthProportion(address _target, bytes _data, uint256 _mul, uint256 _div) external {
        uint256 value = address(this).balance.mul(_mul).div(_div);
         
        require(_target.call.value(value)(_data));
    }

    function approveTokenAmount(address _target, bytes _data, ERC20 _fromToken, uint256 _amount) external {
        if (_fromToken.allowance(this, _target) != 0) {
            _fromToken.asmApprove(_target, 0);
        }
        _fromToken.asmApprove(_target, _amount);
         
        require(_target.call(_data));
    }

    function approveTokenProportion(address _target, bytes _data, ERC20 _fromToken, uint256 _mul, uint256 _div) external {
        uint256 amount = _fromToken.balanceOf(this).mul(_mul).div(_div);
        if (_fromToken.allowance(this, _target) != 0) {
            _fromToken.asmApprove(_target, 0);
        }
        _fromToken.asmApprove(_target, amount);
         
        require(_target.call(_data));
    }

    function transferTokenAmount(address _target, bytes _data, ERC20 _fromToken, uint256 _amount) external {
        _fromToken.asmTransfer(_target, _amount);
        if (_target != address(0)) {
             
            require(_target.call(_data));
        }
    }

    function transferTokenProportion(address _target, bytes _data, ERC20 _fromToken, uint256 _mul, uint256 _div) external {
        uint256 amount = _fromToken.balanceOf(this).mul(_mul).div(_div);
        _fromToken.asmTransfer(_target, amount);
        if (_target != address(0)) {
             
            require(_target.call(_data));
        }
    }

     

    function multitokenChangeAmount(IMultiToken _mtkn, ERC20 _fromToken, ERC20 _toToken, uint256 _minReturn, uint256 _amount) external {
        if (_fromToken.allowance(this, _mtkn) == 0) {
            _fromToken.asmApprove(_mtkn, uint256(-1));
        }
        _mtkn.change(_fromToken, _toToken, _amount, _minReturn);
    }

    function multitokenChangeProportion(IMultiToken _mtkn, ERC20 _fromToken, ERC20 _toToken, uint256 _minReturn, uint256 _mul, uint256 _div) external {
        uint256 amount = _fromToken.balanceOf(this).mul(_mul).div(_div);
        this.multitokenChangeAmount(_mtkn, _fromToken, _toToken, _minReturn, amount);
    }

     

    function withdrawEtherTokenAmount(IEtherToken _etherToken, uint256 _amount) external {
        _etherToken.withdraw(_amount);
    }

    function withdrawEtherTokenProportion(IEtherToken _etherToken, uint256 _mul, uint256 _div) external {
        uint256 amount = _etherToken.balanceOf(this).mul(_mul).div(_div);
        _etherToken.withdraw(amount);
    }

     

    function bancorSendEthValue(IBancorNetwork _bancor, address[] _path, uint256 _value) external {
        _bancor.convert.value(_value)(_path, _value, 1);
    }

    function bancorSendEthProportion(IBancorNetwork _bancor, address[] _path, uint256 _mul, uint256 _div) external {
        uint256 value = address(this).balance.mul(_mul).div(_div);
        _bancor.convert.value(value)(_path, value, 1);
    }

    function bancorApproveTokenAmount(IBancorNetwork _bancor, address[] _path, uint256 _amount) external {
        if (ERC20(_path[0]).allowance(this, _bancor) == 0) {
            ERC20(_path[0]).asmApprove(_bancor, uint256(-1));
        }
        _bancor.claimAndConvert(_path, _amount, 1);
    }

    function bancorApproveTokenProportion(IBancorNetwork _bancor, address[] _path, uint256 _mul, uint256 _div) external {
        uint256 amount = ERC20(_path[0]).balanceOf(this).mul(_mul).div(_div);
        if (ERC20(_path[0]).allowance(this, _bancor) == 0) {
            ERC20(_path[0]).asmApprove(_bancor, uint256(-1));
        }
        _bancor.claimAndConvert(_path, amount, 1);
    }

    function bancorTransferTokenAmount(IBancorNetwork _bancor, address[] _path, uint256 _amount) external {
        ERC20(_path[0]).asmTransfer(_bancor, _amount);
        _bancor.convert(_path, _amount, 1);
    }

    function bancorTransferTokenProportion(IBancorNetwork _bancor, address[] _path, uint256 _mul, uint256 _div) external {
        uint256 amount = ERC20(_path[0]).balanceOf(this).mul(_mul).div(_div);
        ERC20(_path[0]).asmTransfer(_bancor, amount);
        _bancor.convert(_path, amount, 1);
    }

    function bancorAlreadyTransferedTokenAmount(IBancorNetwork _bancor, address[] _path, uint256 _amount) external {
        _bancor.convert(_path, _amount, 1);
    }

    function bancorAlreadyTransferedTokenProportion(IBancorNetwork _bancor, address[] _path, uint256 _mul, uint256 _div) external {
        uint256 amount = ERC20(_path[0]).balanceOf(_bancor).mul(_mul).div(_div);
        _bancor.convert(_path, amount, 1);
    }

     

    function kyberSendEthProportion(IKyberNetworkProxy _kyber, ERC20 _fromToken, address _toToken, uint256 _mul, uint256 _div) external {
        uint256 value = address(this).balance.mul(_mul).div(_div);
        _kyber.trade.value(value)(
            _fromToken,
            value,
            _toToken,
            this,
            1 << 255,
            0,
            0
        );
    }

    function kyberApproveTokenAmount(IKyberNetworkProxy _kyber, ERC20 _fromToken, address _toToken, uint256 _amount) external {
        if (_fromToken.allowance(this, _kyber) == 0) {
            _fromToken.asmApprove(_kyber, uint256(-1));
        }
        _kyber.trade(
            _fromToken,
            _amount,
            _toToken,
            this,
            1 << 255,
            0,
            0
        );
    }

    function kyberApproveTokenProportion(IKyberNetworkProxy _kyber, ERC20 _fromToken, address _toToken, uint256 _mul, uint256 _div) external {
        uint256 amount = _fromToken.balanceOf(this).mul(_mul).div(_div);
        this.kyberApproveTokenAmount(_kyber, _fromToken, _toToken, amount);
    }
}

 

contract MultiSeller is MultiChanger {
    using CheckedERC20 for ERC20;
    using CheckedERC20 for IMultiToken;

    function() public payable {
         
        require(tx.origin != msg.sender);
    }

    function sellForOrigin(
        IMultiToken _mtkn,
        uint256 _amount,
        bytes _callDatas,
        uint[] _starts  
    )
        public
    {
        sell(
            _mtkn,
            _amount,
            _callDatas,
            _starts,
            tx.origin    
        );
    }

    function sell(
        IMultiToken _mtkn,
        uint256 _amount,
        bytes _callDatas,
        uint[] _starts,  
        address _for
    )
        public
    {
        _mtkn.asmTransferFrom(msg.sender, this, _amount);
        _mtkn.unbundle(this, _amount);
        change(_callDatas, _starts);
        _for.transfer(address(this).balance);
    }

     

    function sellOnApproveForOrigin(
        IMultiToken _mtkn,
        uint256 _amount,
        ERC20 _throughToken,
        address[] _exchanges,
        bytes _datas,
        uint[] _datasIndexes  
    )
        public
    {
        sellOnApprove(
            _mtkn,
            _amount,
            _throughToken,
            _exchanges,
            _datas,
            _datasIndexes,
            tx.origin        
        );
    }

    function sellOnApprove(
        IMultiToken _mtkn,
        uint256 _amount,
        ERC20 _throughToken,
        address[] _exchanges,
        bytes _datas,
        uint[] _datasIndexes,  
        address _for
    )
        public
    {
        if (_throughToken == address(0)) {
            require(_mtkn.tokensCount() == _exchanges.length, "sell: _mtkn should have the same tokens count as _exchanges");
        } else {
            require(_mtkn.tokensCount() + 1 == _exchanges.length, "sell: _mtkn should have tokens count + 1 equal _exchanges length");
        }
        require(_datasIndexes.length == _exchanges.length + 1, "sell: _datasIndexes should start with 0 and end with LENGTH");

        _mtkn.transferFrom(msg.sender, this, _amount);
        _mtkn.unbundle(this, _amount);

        for (uint i = 0; i < _exchanges.length; i++) {
            bytes memory data = new bytes(_datasIndexes[i + 1] - _datasIndexes[i]);
            for (uint j = _datasIndexes[i]; j < _datasIndexes[i + 1]; j++) {
                data[j - _datasIndexes[i]] = _datas[j];
            }
            if (data.length == 0) {
                continue;
            }

            if (i == _exchanges.length - 1 && _throughToken != address(0)) {
                if (_throughToken.allowance(this, _exchanges[i]) == 0) {
                    _throughToken.asmApprove(_exchanges[i], uint256(-1));
                }
            } else {
                ERC20 token = _mtkn.tokens(i);
                if (_exchanges[i] == 0) {
                    token.asmTransfer(_for, token.balanceOf(this));
                    continue;
                }
                if (token.allowance(this, _exchanges[i]) == 0) {
                    token.asmApprove(_exchanges[i], uint256(-1));
                }
            }
             
            require(_exchanges[i].call(data), "sell: exchange arbitrary call failed");
        }

        _for.transfer(address(this).balance);
        if (_throughToken != address(0) && _throughToken.balanceOf(this) > 0) {
            _throughToken.asmTransfer(_for, _throughToken.balanceOf(this));
        }
    }
}