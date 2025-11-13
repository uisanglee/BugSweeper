 
pragma solidity ^0.4.24;

 
 

 

 
 
 
 

 
 
 
 

 
 

 

contract DSExec {
    function tryExec( address target, bytes calldata, uint value)
             internal
             returns (bool call_ret)
    {
        return target.call.value(value)(calldata);
    }
    function exec( address target, bytes calldata, uint value)
             internal
    {
        if(!tryExec(target, calldata, value)) {
            revert();
        }
    }

     
    function exec( address t, bytes c )
        internal
    {
        exec(t, c, 0);
    }
    function exec( address t, uint256 v )
        internal
    {
        bytes memory c; exec(t, c, v);
    }
    function tryExec( address t, bytes c )
        internal
        returns (bool)
    {
        return tryExec(t, c, 0);
    }
    function tryExec( address t, uint256 v )
        internal
        returns (bool)
    {
        bytes memory c; return tryExec(t, c, v);
    }
}

 
 

 
 
 
 

 
 
 
 

 
 

 

contract DSNote {
    event LogNote(
        bytes4   indexed  sig,
        address  indexed  guy,
        bytes32  indexed  foo,
        bytes32  indexed  bar,
        uint              wad,
        bytes             fax
    ) anonymous;

    modifier note {
        bytes32 foo;
        bytes32 bar;

        assembly {
            foo := calldataload(4)
            bar := calldataload(36)
        }

        emit LogNote(msg.sig, msg.sender, foo, bar, msg.value, msg.data);

        _;
    }
}

 
 

 

 
 
 
 

 
 
 
 

 
 

 

 
 

contract DaiUpdate is DSExec, DSNote {

    uint256 constant public CAP    = 100000000 * 10 ** 18;  
    address constant public MOM    = 0xF2C5369cFFb8Ea6284452b0326e326DbFdCb867C;  
    address constant public PIP    = 0x40C449c0b74eA531371290115296e7E28b99cf0f;  
    address constant public PEP    = 0x5C1fc813d9c1B5ebb93889B3d63bA24984CA44B7;  
    address constant public MKRUSD = 0x99041F808D598B782D5a3e498681C2452A31da08;  
    address constant public FEED1  = 0xa3E22729A22a8fFEdccBbD614B7430615976E463;  
    address constant public FEED2  = 0x1ec3140C163b6fee00833Ba8ae30A7ba12201063;  

    bool public done;

    function run() public note {
        require(!done);
         
        exec(MOM, abi.encodeWithSignature("setCap(uint256)", CAP), 0);
       
         
        exec(MOM, abi.encodeWithSignature("setPip(address)", PIP), 0);
        
         
        exec(MOM, abi.encodeWithSignature("setPep(address)", PEP), 0);

         
        exec(MKRUSD, abi.encodeWithSignature("set(address)", FEED1), 0);
        exec(MKRUSD, abi.encodeWithSignature("set(address)", FEED2), 0);
        
         
        exec(MKRUSD, abi.encodeWithSignature("setMin(uint96)", 3), 0);

        done = true;
    }
}