Skip to content

Search or jump to…

Pull requests
Issues
Marketplace
Explore
 @SanazME Sign out
1
0 0 ANorwell/expression_parser
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Insights
expression_parser/parser.py
@ANorwell ANorwell Add parser
726c703  10 days ago
87 lines (73 sloc)  2.97 KB

import sys
from collections import namedtuple
import colorama
from colorama import Fore, Back, Style

Val = namedtuple('Val', ['value'])
Op = namedtuple('Op', ['op', 'left', 'right'])

class ParsingError(Exception):
    def __init__(self, message, pos, annotated):
        super().__init__(f"{message} at position {pos}:\n   {annotated}!")
    pass

class Parser:
    def __init__(self, raw):
        self.original = raw

    def parse(self):
        parsed, rest, pos = self.parseHelper(self.original)
        if rest != '':
            self.parsingError(f"Unexpected additional characters: {rest}", pos + 1)
        return parsed

    def parseHelper(self, raw, pos = 0):
        """
         Takes in: a string representing an expression
         Returns: a 3-tuple with:
            1. a parsed object for the first expression in the string
            2. the remaining unparsed string
            3. the current character we have parsed at so far (relative to `self.original`)
        """
        self.ensureNotEmpty(raw, pos)
        triple = self.parseExpr(raw, pos) if (raw[0] == '(') else self.parseNum(raw, pos)
        print(f"parsed {raw} to {triple}")
        return triple

    def parseNum(self, raw, pos):
        return (Val(raw[0]), raw[1:], pos + 1)

    def parseExpr(self, raw, pos):
        left, rest, newPos = self.parseHelper(raw[1:], pos+1)
        self.ensureNotEmpty(rest, newPos)
        op, newPos = self.parseOp(rest[0], newPos)
        right, rest, newPos = self.parseHelper(rest[1:], newPos)
        self.ensureNotEmpty(rest, newPos)
        if rest[0] != ')':
            self.parsingError(f"Unexpected character: {rest[0]}", newPos)
        return (Op(op, left, right), rest[1:], newPos+1)

    def parseOp(self, opChar, pos):
        if opChar not in {'*', '/', '+', '-'}:
            self.parsingError(f"Invalid op {opChar}", pos)
        return (opChar, pos + 1)

    def ensureNotEmpty(self, raw, pos):
        if raw == '':
            self.parsingError(f"Unexpected end of string", pos)

    def parsingError(self, msg, pos):
        if len(self.original) > pos:
            annotated = self.original[0:pos-1] + f"{Back.RED}{self.original[pos]}{Style.RESET_ALL}" + self.original[pos+1:]
        else:
            annotated = self.original[0:pos-1] + f"{Back.RED}[HERE]{Style.RESET_ALL}"
        raise ParsingError(f"{msg} in expression {self.original}", pos, annotated)

class Printer:
    def print(self, node, depth=0):
        if type(node) is Op:
            self.padPrint(node.op, depth)
            self.print(node.left, depth + 1)
            self.print(node.right, depth + 1)
        else:
            self.padPrint(node.value, depth)

    def padPrint(self, string, depth):
        padded = ' ' * (depth*2)
        colors = [Fore.GREEN, Fore.RED, Fore.BLUE]
        color = colors[depth % len(colors)]
        print(f"{padded}{color}{string}{Style.RESET_ALL}")

colorama.init()
try:
    node = Parser(sys.argv[1]).parse()
    Printer().print(node)
except ParsingError as err:
    print(str(err))
© 2019 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
