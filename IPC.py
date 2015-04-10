class IPC:
    """
    A identifier describing a patent classification symbol. Symbols are of the
    form X00Yn/m; one letter (section), two digits (class), one letter (subclass) 
    and two numbers separated by a slash (group). Symbols are organized into
    a hierarchy, with the levels going section > class > subclass > group.
    An example with IPC H04N3/38:
        H (section) is "Electricity"
        H04 (class) is "Electric communication techniques"
        H04N (subclass) is "Pictorial communication, e.g. television"
        H04N3/38 (group) is "Scanning of motion picture films with continously moving film"
    One important thing to note is that groups can also be nested. In the example above,
    3/38 is a subgroup of 3/36 which is "Scanning of motion picture films".
    
    This class supports partial IPC symbols, and checking whether an IPC symbol is included
    in another (e.g. H04 is in H). It does not support checking membership for subgroups
    (e.g. it will say that H04N3/38 is not in H04N3/36).
    
    An IPC object constructed with an empty string will contain any other object except
    those also constructed with empty strings.
    """
    
    def __init__(self, symbol):
        """
        Create an IPC symbol object from a string representing the symbol. Some
        validations are performed to ensure the symbol is correct.
        """
        symbol = symbol.rstrip().lstrip()
        
        self._section = self._class = self._subclass = self._group = ''
        self._symbol = symbol
        
        # IPC is empty, nothing else to do
        if len(symbol) == 0:
            return
        
        # Check that the section is correct
        if symbol[0] not in 'ABCDEFGHabcdefgh':
            raise RuntimeError, 'IPC: invalid patent section'
        self._section = symbol[0]
        
        # If we only have a section, stop here
        if len(symbol) == 1:
            return
        
        if len(symbol) == 2:
            raise RuntimeError, 'IPC: patent class should have two digits'
        
        if not symbol[1:3].isdigit():
            raise RuntimeError, 'IPC: invalid patent class (not a number)'
            
        self._class = symbol[1:3]
        
        # Stop if we don't have anything more than a class
        if len(symbol) == 3:
            return
        
        if not symbol[3].isalpha():
            raise RuntimeError, 'IPC: invalid patent subclass (not a letter)'
        self._subclass = symbol[3]
        
        if len(symbol) == 4:
            return
        
        # Some patents in the corpus have no group, but still include the slash
        if len(symbol) == 5 and symbol[4] == '/':
            return
        
        group = symbol[4:]
        group_nums = group.split('/')
        if len(group_nums) != 2:
            raise RuntimeError, ('IPC: invalid patent group %s (format must be x/y with x and y numbers)' % group)
            
        if not group_nums[0].isdigit() or not group_nums[1].isdigit():
            raise RuntimeError, ('IPC: invalid patent group %s (format must be x/y with x and y numbers)' % group)
            
        self._group = group
    
    def __eq__(self, other):
        """
        Check whether this symbol is equal to another one. Two symbols are equal if and only
        if their textual representation is the same.
        """
        if not isinstance(other, IPC):
            return False
        return self._symbol == other._symbol
        
    def __str__(self):
        """
        Return the string describing this symbol
        """
        return self._symbol
    
    def __hash__(self):
        """
        Return the hash of this symbol.
        """
        return hash(self._symbol)
    
    def __contains__(self, other):
        """
        Check whether this symbol contains another symbol. Note that identical symbols
        do not contain each other (e.g. IPC("H") in IPC("H") == False).
        """
    
        if not isinstance(other, IPC):
            return False
        # If we are an empty symbol, we contain the other unless they also are an empty symbol
        if len(self._symbol) == 0:
            return len(other._symbol) != 0
        
        # If the sections differ, then we don't contain them
        if self._section != other._section:
            return False
        
        # The sections are identical. If we have no class, we contain them only if they do have a class
        # (e.g. "H" in "H" == False)
        if len(self._class) == 0:
            return len(other._class) != 0
            
        # We do have a class. If it's not the same as the other, we don't contain them.
        if self._class != other._class:
            return False
            
        # The classes are identical. If we have no subclass, we contain them only if they do have a subclass
        if len(self._subclass) == 0:
            return len(other._subclass) != 0
            
        # We do have a subclass. If it's not the same as the other, we don't contain them.
        if self._subclass != other._subclass:
            return False
            
        # The subclasses are identical. If we have no group, we contain them only if they have a group
        if len(self._group) == 0:
            return len(other._group) != 0
            
        # We have a group. Therefore we can't contain anything.
        return False

    def getPatents(self, patent_info):
        """
        Return all patentNo that are part of the subclass of the current IPC
        """
        patentNos = []

        POSITION_IPC = 2

        for patentNo in patent_info:
            ipc = IPC(patent_info[patentNo][POSITION_IPC])
            
            if ipc in self:
                patentNos.append(patentNo)

        return patentNos
        
    def section(self):
        """
        Return a new IPC symbol comprising of just the section of this symbol
        """
        return IPC(self._section)
        
    def mainClass(self):
        """
        Return a new IPC symbol made out of the section and class of this symbol
        """
        return IPC(self._section + self._class)
        
    def subclass(self):
        """
        Return a new IPC symbol made out of the section, class and subclass of this symbol
        """
        return IPC(self._section + self._class + self._subclass)