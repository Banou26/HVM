// HVM2 WebGPU Compute Shader
// ===========================
// Implements interaction net evaluation on WebGPU

// Constants
// ---------

const VAR: u32 = 0u; // variable
const REF: u32 = 1u; // reference
const ERA: u32 = 2u; // eraser
const NUM: u32 = 3u; // number
const CON: u32 = 4u; // constructor
const DUP: u32 = 5u; // duplicator
const OPR: u32 = 6u; // operator
const SWI: u32 = 7u; // switch

// Rules
const LINK: u32 = 0u;
const CALL: u32 = 1u;
const VOID: u32 = 2u;
const ERAS: u32 = 3u;
const ANNI: u32 = 4u;
const COMM: u32 = 5u;
const OPER: u32 = 6u;
const SWIT: u32 = 7u;

// Number types
const TY_SYM: u32 = 0u;
const TY_U24: u32 = 1u;
const TY_I24: u32 = 2u;
const TY_F24: u32 = 3u;

// Number operations
const OP_ADD: u32 = 4u;
const OP_SUB: u32 = 5u;
const FP_SUB: u32 = 6u;
const OP_MUL: u32 = 7u;
const OP_DIV: u32 = 8u;
const FP_DIV: u32 = 9u;
const OP_REM: u32 = 10u;
const FP_REM: u32 = 11u;
const OP_EQ: u32 = 12u;
const OP_NEQ: u32 = 13u;
const OP_LT: u32 = 14u;
const OP_GT: u32 = 15u;
const OP_AND: u32 = 16u;
const OP_OR: u32 = 17u;
const OP_XOR: u32 = 18u;
const OP_SHL: u32 = 19u;
const FP_SHL: u32 = 20u;
const OP_SHR: u32 = 21u;
const FP_SHR: u32 = 22u;

// Special ports
const FREE: u32 = 0x00000000u;
const ROOT: u32 = 0xFFFFFFF8u;
const NONE: u32 = 0xFFFFFFFFu;

// Configuration
const WORKGROUP_SIZE: u32 = 256u;
const RLEN: u32 = 256u; // redex bag length per thread
const L_NODE_LEN: u32 = 0x2000u;
const L_VARS_LEN: u32 = 0x2000u;

// Data Structures
// ---------------

struct NetConfig {
  node_len: u32,
  vars_len: u32,
  rbag_len: u32,
  defs_len: u32,
  mode: u32,
  done: atomic<u32>,
  itrs: atomic<u32>,
  rwts: atomic<u32>,
}

// Buffers
// -------

@group(0) @binding(0) var<storage, read_write> node_buf: array<atomic<u32>>; // Pairs stored as 2 u32s
@group(0) @binding(1) var<storage, read_write> vars_buf: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> rbag_buf: array<u32>; // Redex bag (pairs)
@group(0) @binding(3) var<storage, read_write> rbag_len: array<atomic<u32>>; // Per-thread redex counts
@group(0) @binding(4) var<storage, read> book_buf: array<u32>; // Book of definitions
@group(0) @binding(5) var<storage, read_write> config: NetConfig;

// Workgroup shared memory for local redex bag
var<workgroup> local_hi: array<u32, 512>; // High priority redexes (pairs as 2 u32s)
var<workgroup> local_lo: array<u32, 512>; // Low priority redexes
var<workgroup> hi_end: atomic<u32>;
var<workgroup> lo_end: atomic<u32>;

// Port/Pair Operations
// --------------------

fn new_port(tag: u32, val: u32) -> u32 {
  return (val << 3u) | tag;
}

fn get_tag(port: u32) -> u32 {
  return port & 7u;
}

fn get_val(port: u32) -> u32 {
  return port >> 3u;
}

fn is_nod(port: u32) -> bool {
  return get_tag(port) >= CON;
}

fn is_var(port: u32) -> bool {
  return get_tag(port) == VAR;
}

fn get_fst_idx(loc: u32) -> u32 {
  return loc * 2u;
}

fn get_snd_idx(loc: u32) -> u32 {
  return loc * 2u + 1u;
}

// Rule lookup table
fn get_rule(a: u32, b: u32) -> u32 {
  let tag_a = get_tag(a);
  let tag_b = get_tag(b);

  // Rule table: [a_tag][b_tag] -> rule
  // VAR -> LINK with everything
  if tag_a == VAR || tag_b == VAR {
    return LINK;
  }

  // REF interactions
  if tag_a == REF {
    if tag_b == REF || tag_b == ERA || tag_b == NUM {
      return VOID;
    }
    return CALL;
  }
  if tag_b == REF {
    if tag_a == ERA || tag_a == NUM {
      return VOID;
    }
    return CALL;
  }

  // ERA interactions
  if tag_a == ERA {
    if tag_b == ERA || tag_b == NUM {
      return VOID;
    }
    return ERAS;
  }
  if tag_b == ERA {
    if tag_a == NUM {
      return VOID;
    }
    return ERAS;
  }

  // NUM interactions
  if tag_a == NUM {
    if tag_b == NUM {
      return VOID;
    }
    if tag_b == OPR {
      return OPER;
    }
    if tag_b == SWI {
      return SWIT;
    }
    return ERAS;
  }
  if tag_b == NUM {
    if tag_a == OPR {
      return OPER;
    }
    if tag_a == SWI {
      return SWIT;
    }
    return ERAS;
  }

  // Node-Node interactions
  if tag_a == tag_b {
    return ANNI;
  }
  return COMM;
}

fn should_swap(a: u32, b: u32) -> bool {
  return get_tag(b) < get_tag(a);
}

fn is_high_priority(rule: u32) -> bool {
  // LINK, CALL, VOID, ERAS, SWIT are high priority (bits: 00011101)
  return ((0x1Du >> rule) & 1u) != 0u;
}

// Number operations
// -----------------

fn new_u24(val: u32) -> u32 {
  return (val << 5u) | TY_U24;
}

fn get_u24(word: u32) -> u32 {
  return word >> 5u;
}

fn new_i24(val: i32) -> u32 {
  return (u32(val) << 5u) | TY_I24;
}

fn get_i24(word: u32) -> i32 {
  return (i32(word) << 3) >> 8;
}

fn get_typ(word: u32) -> u32 {
  return word & 0x1Fu;
}

fn get_sym(word: u32) -> u32 {
  return word >> 5u;
}

fn new_sym(val: u32) -> u32 {
  return (val << 5u) | TY_SYM;
}

fn operate_u24(op: u32, av: u32, bv: u32) -> u32 {
  switch op {
    case 4u: { return new_u24((av + bv) & 0xFFFFFFu); } // ADD
    case 5u: { return new_u24((av - bv) & 0xFFFFFFu); } // SUB
    case 6u: { return new_u24((bv - av) & 0xFFFFFFu); } // FP_SUB
    case 7u: { return new_u24((av * bv) & 0xFFFFFFu); } // MUL
    case 8u: { if bv == 0u { return new_u24(0u); } return new_u24(av / bv); } // DIV
    case 9u: { if av == 0u { return new_u24(0u); } return new_u24(bv / av); } // FP_DIV
    case 10u: { if bv == 0u { return new_u24(0u); } return new_u24(av % bv); } // REM
    case 11u: { if av == 0u { return new_u24(0u); } return new_u24(bv % av); } // FP_REM
    case 12u: { return new_u24(select(0u, 1u, av == bv)); } // EQ
    case 13u: { return new_u24(select(0u, 1u, av != bv)); } // NEQ
    case 14u: { return new_u24(select(0u, 1u, av < bv)); } // LT
    case 15u: { return new_u24(select(0u, 1u, av > bv)); } // GT
    case 16u: { return new_u24(av & bv); } // AND
    case 17u: { return new_u24(av | bv); } // OR
    case 18u: { return new_u24(av ^ bv); } // XOR
    case 19u: { return new_u24((av << (bv & 31u)) & 0xFFFFFFu); } // SHL
    case 20u: { return new_u24((bv << (av & 31u)) & 0xFFFFFFu); } // FP_SHL
    case 21u: { return new_u24(av >> (bv & 31u)); } // SHR
    case 22u: { return new_u24(bv >> (av & 31u)); } // FP_SHR
    default: { return new_u24(0u); }
  }
}

fn operate_i24(op: u32, av: i32, bv: i32) -> u32 {
  switch op {
    case 4u: { return new_i24(av + bv); } // ADD
    case 5u: { return new_i24(av - bv); } // SUB
    case 6u: { return new_i24(bv - av); } // FP_SUB
    case 7u: { return new_i24(av * bv); } // MUL
    case 8u: { if bv == 0 { return new_i24(0); } return new_i24(av / bv); } // DIV
    case 9u: { if av == 0 { return new_i24(0); } return new_i24(bv / av); } // FP_DIV
    case 10u: { if bv == 0 { return new_i24(0); } return new_i24(av % bv); } // REM
    case 11u: { if av == 0 { return new_i24(0); } return new_i24(bv % av); } // FP_REM
    case 12u: { return new_u24(select(0u, 1u, av == bv)); } // EQ
    case 13u: { return new_u24(select(0u, 1u, av != bv)); } // NEQ
    case 14u: { return new_u24(select(0u, 1u, av < bv)); } // LT
    case 15u: { return new_u24(select(0u, 1u, av > bv)); } // GT
    case 16u: { return new_i24(av & bv); } // AND
    case 17u: { return new_i24(av | bv); } // OR
    case 18u: { return new_i24(av ^ bv); } // XOR
    default: { return new_i24(0); }
  }
}

fn operate(a: u32, b: u32) -> u32 {
  let at = get_typ(a);
  let bt = get_typ(b);

  // Both symbols
  if at == TY_SYM && bt == TY_SYM {
    return new_u24(0u);
  }

  // One symbol = partial application
  if at == TY_SYM && bt != TY_SYM {
    return (b & ~0x1Fu) | get_sym(a);
  }
  if at != TY_SYM && bt == TY_SYM {
    return (a & ~0x1Fu) | get_sym(b);
  }

  // Both are operations
  if at >= OP_ADD && bt >= OP_ADD {
    return new_u24(0u);
  }

  // Both are values
  if at < OP_ADD && bt < OP_ADD {
    return new_u24(0u);
  }

  // One operation, one value
  var op: u32;
  var aval: u32;
  var ty: u32;
  var bval: u32;

  if at >= OP_ADD {
    op = at;
    aval = a;
    ty = bt;
    bval = b;
  } else {
    op = bt;
    aval = b;
    ty = at;
    bval = a;
  }

  if ty == TY_U24 {
    return operate_u24(op, get_u24(aval), get_u24(bval));
  } else if ty == TY_I24 {
    return operate_i24(op, get_i24(aval), get_i24(bval));
  }
  // F24 operations would need more complex handling
  return new_u24(0u);
}

// Memory Operations
// -----------------

fn node_load_fst(loc: u32) -> u32 {
  return atomicLoad(&node_buf[get_fst_idx(loc)]);
}

fn node_load_snd(loc: u32) -> u32 {
  return atomicLoad(&node_buf[get_snd_idx(loc)]);
}

fn node_store(loc: u32, fst: u32, snd: u32) {
  atomicStore(&node_buf[get_fst_idx(loc)], fst);
  atomicStore(&node_buf[get_snd_idx(loc)], snd);
}

fn node_create(loc: u32, fst: u32, snd: u32) {
  atomicStore(&node_buf[get_fst_idx(loc)], fst);
  atomicStore(&node_buf[get_snd_idx(loc)], snd);
}

fn node_take(loc: u32) -> vec2<u32> {
  let fst = atomicExchange(&node_buf[get_fst_idx(loc)], 0u);
  let snd = atomicExchange(&node_buf[get_snd_idx(loc)], 0u);
  return vec2<u32>(fst, snd);
}

fn node_exchange(loc: u32, fst: u32, snd: u32) -> vec2<u32> {
  let old_fst = atomicExchange(&node_buf[get_fst_idx(loc)], fst);
  let old_snd = atomicExchange(&node_buf[get_snd_idx(loc)], snd);
  return vec2<u32>(old_fst, old_snd);
}

fn vars_load(var_idx: u32) -> u32 {
  return atomicLoad(&vars_buf[var_idx]);
}

fn vars_store(var_idx: u32, val: u32) {
  atomicStore(&vars_buf[var_idx], val);
}

fn vars_create(var_idx: u32, val: u32) {
  atomicStore(&vars_buf[var_idx], val);
}

fn vars_take(var_idx: u32) -> u32 {
  return atomicExchange(&vars_buf[var_idx], 0u);
}

fn vars_exchange(var_idx: u32, val: u32) -> u32 {
  return atomicExchange(&vars_buf[var_idx], val);
}

fn is_node_free(loc: u32) -> bool {
  return node_load_fst(loc) == 0u && node_load_snd(loc) == 0u;
}

fn is_vars_free(var_idx: u32) -> bool {
  return vars_load(var_idx) == 0u;
}

// Enter: follows variable substitution chain
fn enter(port: u32) -> u32 {
  var p = port;
  while get_tag(p) == VAR {
    let val = vars_exchange(get_val(p), NONE);
    if val == NONE || val == 0u {
      break;
    }
    vars_take(get_val(p));
    p = val;
  }
  return p;
}

// Book Operations
// ---------------

// Read a definition from the book buffer
// Book format:
// - defs_len: u32
// - For each def:
//   - fid: u32
//   - name: 256 bytes (64 u32s)
//   - safe: u32
//   - rbag_len: u32
//   - node_len: u32
//   - vars_len: u32
//   - root: u32
//   - rbag_buf: rbag_len * 2 u32s
//   - node_buf: node_len * 2 u32s

struct DefHeader {
  safe: u32,
  rbag_len: u32,
  node_len: u32,
  vars_len: u32,
  root: u32,
  rbag_offset: u32,
  node_offset: u32,
}

fn read_def_header(fid: u32) -> DefHeader {
  // Skip to the right definition
  var offset = 1u; // Skip defs_len

  for (var i = 0u; i < fid; i++) {
    offset += 1u; // fid
    offset += 64u; // name (256 bytes = 64 u32s)
    let safe = book_buf[offset]; offset += 1u;
    let rbag_len = book_buf[offset]; offset += 1u;
    let node_len = book_buf[offset]; offset += 1u;
    let vars_len = book_buf[offset]; offset += 1u;
    offset += 1u; // root
    offset += rbag_len * 2u; // rbag_buf
    offset += node_len * 2u; // node_buf
  }

  offset += 1u; // fid
  offset += 64u; // name

  var header: DefHeader;
  header.safe = book_buf[offset]; offset += 1u;
  header.rbag_len = book_buf[offset]; offset += 1u;
  header.node_len = book_buf[offset]; offset += 1u;
  header.vars_len = book_buf[offset]; offset += 1u;
  header.root = book_buf[offset]; offset += 1u;
  header.rbag_offset = offset;
  header.node_offset = offset + header.rbag_len * 2u;

  return header;
}

fn read_def_rbag(header: DefHeader, idx: u32) -> vec2<u32> {
  let base = header.rbag_offset + idx * 2u;
  return vec2<u32>(book_buf[base], book_buf[base + 1u]);
}

fn read_def_node(header: DefHeader, idx: u32) -> vec2<u32> {
  let base = header.node_offset + idx * 2u;
  return vec2<u32>(book_buf[base], book_buf[base + 1u]);
}

// Allocation
// ----------

var<private> nput: u32;
var<private> vput: u32;
var<private> nloc: array<u32, 256>;
var<private> vloc: array<u32, 256>;

fn node_alloc(num: u32, gid: u32) -> u32 {
  var got = 0u;
  let node_len = config.node_len;
  let start = (gid * 4096u) % node_len;

  for (var i = 0u; i < node_len && got < num; i++) {
    nput = (start + nput + 1u) % node_len;
    if nput > 0u && is_node_free(nput) {
      nloc[got] = nput;
      got++;
    }
  }
  return got;
}

fn vars_alloc(num: u32, gid: u32) -> u32 {
  var got = 0u;
  let vars_len = config.vars_len;
  let start = (gid * 4096u) % vars_len;

  for (var i = 0u; i < vars_len && got < num; i++) {
    vput = (start + vput + 1u) % vars_len;
    if vput > 0u && is_vars_free(vput) {
      vloc[got] = vput;
      got++;
    }
  }
  return got;
}

// Adjust port based on allocated locations
fn adjust_port(port: u32) -> u32 {
  let tag = get_tag(port);
  let val = get_val(port);
  if is_nod(port) {
    return new_port(tag, nloc[val]);
  }
  if is_var(port) {
    return new_port(tag, vloc[val]);
  }
  return port;
}

// Interaction Implementations
// ---------------------------

// Redex bag operations
var<private> pri_hi: array<vec2<u32>, 256>;
var<private> pri_lo: array<vec2<u32>, 256>;
var<private> hi_len: u32;
var<private> lo_len: u32;

fn push_redex(a: u32, b: u32) {
  let rule = get_rule(a, b);
  if is_high_priority(rule) {
    if hi_len < 256u {
      pri_hi[hi_len] = vec2<u32>(a, b);
      hi_len++;
    }
  } else {
    if lo_len < 256u {
      pri_lo[lo_len] = vec2<u32>(a, b);
      lo_len++;
    }
  }
}

fn pop_redex() -> vec2<u32> {
  if hi_len > 0u {
    hi_len--;
    return pri_hi[hi_len];
  }
  if lo_len > 0u {
    lo_len--;
    return pri_lo[lo_len];
  }
  return vec2<u32>(NONE, NONE);
}

fn has_redex() -> bool {
  return hi_len > 0u || lo_len > 0u;
}

// Link operation
fn link(a_in: u32, b_in: u32) {
  var a = a_in;
  var b = b_in;

  loop {
    // If A is NODE and B is VAR: swap
    if get_tag(a) != VAR && get_tag(b) == VAR {
      let tmp = a; a = b; b = tmp;
    }

    // If A is NODE: create redex
    if get_tag(a) != VAR {
      push_redex(a, b);
      break;
    }

    // Extend B
    b = enter(b);

    // A is VAR: store A -> B
    let a_old = vars_exchange(get_val(a), b);
    if a_old == NONE {
      break;
    }
    vars_take(get_val(a));
    a = a_old;
  }
}

// LINK interaction
fn interact_link(a: u32, b: u32) -> bool {
  link(a, b);
  return true;
}

// CALL interaction
fn interact_call(a: u32, b: u32, gid: u32) -> bool {
  let fid = get_val(a) & 0xFFFFFFFu;
  let header = read_def_header(fid);

  // Copy optimization for DUP
  if get_tag(b) == DUP && header.safe != 0u {
    return interact_eras(a, b, gid);
  }

  // Allocate resources
  let got_nodes = node_alloc(header.node_len, gid);
  let got_vars = vars_alloc(header.vars_len, gid);

  if got_nodes < header.node_len || got_vars < header.vars_len {
    return false;
  }

  // Create vars
  for (var i = 0u; i < header.vars_len; i++) {
    vars_create(vloc[i], NONE);
  }

  // Create nodes
  for (var i = 0u; i < header.node_len; i++) {
    let pair = read_def_node(header, i);
    node_create(nloc[i], adjust_port(pair.x), adjust_port(pair.y));
  }

  // Link rbag
  for (var i = 0u; i < header.rbag_len; i++) {
    let pair = read_def_rbag(header, i);
    link(adjust_port(pair.x), adjust_port(pair.y));
  }

  // Link root
  link(adjust_port(header.root), b);

  return true;
}

// VOID interaction
fn interact_void(a: u32, b: u32) -> bool {
  return true;
}

// ERAS interaction
fn interact_eras(a: u32, b: u32, gid: u32) -> bool {
  let b_loc = get_val(b);

  // Check availability
  if is_node_free(b_loc) {
    return false;
  }

  let b_pair = node_exchange(b_loc, 0u, 0u);
  let b1 = b_pair.x;
  let b2 = b_pair.y;

  link(a, b1);
  link(a, b2);

  return true;
}

// ANNI interaction
fn interact_anni(a: u32, b: u32, gid: u32) -> bool {
  let a_loc = get_val(a);
  let b_loc = get_val(b);

  // Check availability
  if is_node_free(a_loc) || is_node_free(b_loc) {
    return false;
  }

  let a_pair = node_take(a_loc);
  let b_pair = node_take(b_loc);

  link(a_pair.x, b_pair.x);
  link(a_pair.y, b_pair.y);

  return true;
}

// COMM interaction
fn interact_comm(a: u32, b: u32, gid: u32) -> bool {
  // Allocate 4 nodes and 4 vars
  let got_nodes = node_alloc(4u, gid);
  let got_vars = vars_alloc(4u, gid);

  if got_nodes < 4u || got_vars < 4u {
    return false;
  }

  let a_loc = get_val(a);
  let b_loc = get_val(b);

  // Check availability
  if is_node_free(a_loc) || is_node_free(b_loc) {
    return false;
  }

  let a_pair = node_take(a_loc);
  let b_pair = node_take(b_loc);

  // Create vars
  vars_create(vloc[0], NONE);
  vars_create(vloc[1], NONE);
  vars_create(vloc[2], NONE);
  vars_create(vloc[3], NONE);

  // Create nodes
  node_create(nloc[0], new_port(VAR, vloc[0]), new_port(VAR, vloc[1]));
  node_create(nloc[1], new_port(VAR, vloc[2]), new_port(VAR, vloc[3]));
  node_create(nloc[2], new_port(VAR, vloc[0]), new_port(VAR, vloc[2]));
  node_create(nloc[3], new_port(VAR, vloc[1]), new_port(VAR, vloc[3]));

  // Links
  link(new_port(get_tag(b), nloc[0]), a_pair.x);
  link(new_port(get_tag(b), nloc[1]), a_pair.y);
  link(new_port(get_tag(a), nloc[2]), b_pair.x);
  link(new_port(get_tag(a), nloc[3]), b_pair.y);

  return true;
}

// OPER interaction
fn interact_oper(a: u32, b: u32, gid: u32) -> bool {
  let b_loc = get_val(b);

  // Check availability
  if is_node_free(b_loc) {
    return false;
  }

  let av = get_val(a);
  let b_pair = node_take(b_loc);
  var b1 = b_pair.x;
  let b2 = enter(b_pair.y);

  if get_tag(b1) == NUM {
    let bv = get_val(b1);
    let cv = operate(av, bv);
    link(new_port(NUM, cv), b2);
  } else {
    let got_nodes = node_alloc(1u, gid);
    if got_nodes < 1u {
      return false;
    }
    node_create(nloc[0], new_port(NUM, av), b2);
    link(b1, new_port(OPR, nloc[0]));
  }

  return true;
}

// SWIT interaction
fn interact_swit(a: u32, b: u32, gid: u32) -> bool {
  let b_loc = get_val(b);

  // Check availability
  if is_node_free(b_loc) {
    return false;
  }

  let got_nodes = node_alloc(2u, gid);
  if got_nodes < 2u {
    return false;
  }

  let av = get_u24(get_val(a));
  let b_pair = node_take(b_loc);
  let b1 = b_pair.x;
  let b2 = b_pair.y;

  if av == 0u {
    node_create(nloc[0], b2, new_port(ERA, 0u));
    link(new_port(CON, nloc[0]), b1);
  } else {
    node_create(nloc[0], new_port(ERA, 0u), new_port(CON, nloc[1]));
    node_create(nloc[1], new_port(NUM, new_u24(av - 1u)), b2);
    link(new_port(CON, nloc[0]), b1);
  }

  return true;
}

// Main interaction dispatcher
fn interact(a_in: u32, b_in: u32, gid: u32) -> bool {
  var a = a_in;
  var b = b_in;

  var rule = get_rule(a, b);

  // Handle root redex
  if get_tag(a) == REF && b == ROOT {
    rule = CALL;
  } else if should_swap(a, b) {
    let tmp = a; a = b; b = tmp;
  }

  var success = false;
  switch rule {
    case 0u: { success = interact_link(a, b); }
    case 1u: { success = interact_call(a, b, gid); }
    case 2u: { success = interact_void(a, b); }
    case 3u: { success = interact_eras(a, b, gid); }
    case 4u: { success = interact_anni(a, b, gid); }
    case 5u: { success = interact_comm(a, b, gid); }
    case 6u: { success = interact_oper(a, b, gid); }
    case 7u: { success = interact_swit(a, b, gid); }
    default: { success = false; }
  }

  return success && rule != LINK;
}

// Main Compute Kernel
// -------------------

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let gid = global_id.x;

  // Initialize private state
  hi_len = 0u;
  lo_len = 0u;
  nput = gid * 128u;
  vput = gid * 128u;

  // Load initial redexes from global bag
  let rbag_start = gid * RLEN;
  let my_len = atomicLoad(&rbag_len[gid]);

  for (var i = 0u; i < my_len && i < RLEN; i++) {
    let idx = (rbag_start + i) * 2u;
    let a = rbag_buf[idx];
    let b = rbag_buf[idx + 1u];
    push_redex(a, b);
  }

  // Clear global count
  atomicStore(&rbag_len[gid], 0u);

  var itrs = 0u;
  let max_itrs = 65536u;

  // Main evaluation loop
  while has_redex() && itrs < max_itrs {
    let redex = pop_redex();
    if redex.x == NONE {
      break;
    }

    if interact(redex.x, redex.y, gid) {
      itrs++;
    } else {
      // Failed, push back
      push_redex(redex.x, redex.y);
      break;
    }
  }

  // Store remaining redexes back to global
  var store_idx = 0u;
  for (var i = 0u; i < hi_len; i++) {
    let idx = (rbag_start + store_idx) * 2u;
    rbag_buf[idx] = pri_hi[i].x;
    rbag_buf[idx + 1u] = pri_hi[i].y;
    store_idx++;
  }
  for (var i = 0u; i < lo_len; i++) {
    let idx = (rbag_start + store_idx) * 2u;
    rbag_buf[idx] = pri_lo[i].x;
    rbag_buf[idx + 1u] = pri_lo[i].y;
    store_idx++;
  }
  atomicStore(&rbag_len[gid], store_idx);

  // Update global counters
  atomicAdd(&config.itrs, itrs);

  if store_idx == 0u {
    atomicAdd(&config.done, 1u);
  }
}

// Boot kernel - initializes the first redex
@compute @workgroup_size(1)
fn boot(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // Find main function ID
  let defs_len = book_buf[0];
  var main_fid = 0u;

  var offset = 1u;
  for (var i = 0u; i < defs_len; i++) {
    offset += 1u; // fid
    // Check if name is "main"
    let c0 = book_buf[offset] & 0xFFu;
    let c1 = (book_buf[offset] >> 8u) & 0xFFu;
    let c2 = (book_buf[offset] >> 16u) & 0xFFu;
    let c3 = (book_buf[offset] >> 24u) & 0xFFu;

    if c0 == 109u && c1 == 97u && c2 == 105u && c3 == 110u { // 'm', 'a', 'i', 'n'
      main_fid = i;
    }

    offset += 64u; // name
    offset += 1u; // safe
    let rbag_len_val = book_buf[offset]; offset += 1u;
    let node_len_val = book_buf[offset]; offset += 1u;
    offset += 1u; // vars_len
    offset += 1u; // root
    offset += rbag_len_val * 2u;
    offset += node_len_val * 2u;
  }

  // Create root variable
  vars_store(0x1FFFFFFFu, NONE);

  // Push initial redex: @main ~ ROOT
  rbag_buf[0] = new_port(REF, main_fid);
  rbag_buf[1] = ROOT;
  atomicStore(&rbag_len[0], 1u);

  atomicStore(&config.done, 0u);
  atomicStore(&config.itrs, 0u);
}
