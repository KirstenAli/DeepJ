function fixture(name, body, cognitive, nesting) {
  return { name, source: `int ${name}(int x) { ${body} return x; }`, cognitive, nesting };
}

export const fixtures = [
  fixture('boundary15', 'if (x > 0) { x--; } '.repeat(15), 15, 1),
  fixture('boundary16', 'if (x > 0) { x--; } '.repeat(16), 16, 1),
  fixture('threeLevels', 'if (x) { for (int i=0;i<x;i++) { while (x) { x--; } } }', 6, 3),
  fixture(
    'fourLevels',
    'if (x) { for (int i=0;i<x;i++) { while (x) { if (x) { x--; } } } }',
    10,
    4,
  ),
  fixture('unbraced', 'if (x) if (x > 1) if (x > 2) if (x > 3) x--;', 10, 4),
  fixture('elseChain', 'if (x) { x--; } ' + 'else if (x > 1) { x--; } '.repeat(5), 6, 1),
  fixture('elseBranch', 'if (x) { x--; } else { if (x > 1) { x++; } }', 4, 2),
  fixture('ternary', 'x = x > 0 ? 1 : 2;', 1, 0),
  fixture('nestedTernary', 'x = x > 0 ? (x > 1 ? 1 : 2) : 3;', 3, 0),
  fixture('gnuTernary', 'x = (x > 0 && x < 2) ?: 1;', 2, 0),
  fixture(
    'switchCases',
    'switch (x) { case 1: x++; break; case 2: x--; break; default: break; }',
    1,
    1,
  ),
  fixture('doLoop', 'do { x--; } while (x > 0);', 1, 1),
  fixture(
    'rangeLoop',
    'int values[] = {1, 2}; for (int value : values) { if (value) { x--; } }',
    3,
    2,
  ),
  fixture('catchBranch', 'try { x--; } catch (...) { if (x) { x++; } }', 3, 2),
  fixture('sameLogical', 'x = x > 0 && x < 2 && x != 1;', 1, 0),
  fixture('mixedLogical', 'x = x > 0 && (x < 2 || x != 1);', 2, 0),
  fixture('repeatedMixedLogical', 'x = (x > 0 || x < 2) && (x != 1 || x > 3);', 3, 0),
  fixture('commentText', '/* if (x) { for (;;) { while (x) { */ x--;', 0, 0),
  fixture('stringText', 'const char* text = "if (x) { for (;;) { while (x) {";', 0, 0),
  fixture('lambdaBody', 'auto fn = [x] { if (x) { return 1; } return 0; }; x = fn();', 2, 1),
  fixture('recursive', 'if (x > 0) { return recursive(x - 1); }', 2, 1),
];
