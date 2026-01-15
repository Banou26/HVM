/**
 * HVM WebGPU Runtime
 *
 * A JavaScript/WebGPU implementation of the HVM2 interaction net evaluator.
 * This can run HVM programs compiled from Bend or written directly in HVM IR.
 */

// WGSL Shader source
export const HVM_SHADER = `
// HVM2 WebGPU Compute Shader
// ===========================

// Constants
const VAR: u32 = 0u;
const REF: u32 = 1u;
const ERA: u32 = 2u;
const NUM: u32 = 3u;
const CON: u32 = 4u;
const DUP: u32 = 5u;
const OPR: u32 = 6u;
const SWI: u32 = 7u;

const LINK: u32 = 0u;
const CALL: u32 = 1u;
const VOID: u32 = 2u;
const ERAS: u32 = 3u;
const ANNI: u32 = 4u;
const COMM: u32 = 5u;
const OPER: u32 = 6u;
const SWIT: u32 = 7u;

const TY_SYM: u32 = 0u;
const TY_U24: u32 = 1u;
const TY_I24: u32 = 2u;
const TY_F24: u32 = 3u;
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

const FREE: u32 = 0x00000000u;
const ROOT: u32 = 0xFFFFFFF8u;
const NONE: u32 = 0xFFFFFFFFu;
const RLEN: u32 = 256u;

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

@group(0) @binding(0) var<storage, read_write> node_buf: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> vars_buf: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> rbag_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> rbag_len: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> book_buf: array<u32>;
@group(0) @binding(5) var<uniform> config: NetConfig;

fn new_port(tag: u32, val: u32) -> u32 { return (val << 3u) | tag; }
fn get_tag(port: u32) -> u32 { return port & 7u; }
fn get_val(port: u32) -> u32 { return port >> 3u; }
fn is_nod(port: u32) -> bool { return get_tag(port) >= CON; }
fn is_var(port: u32) -> bool { return get_tag(port) == VAR; }
fn get_fst_idx(loc: u32) -> u32 { return loc * 2u; }
fn get_snd_idx(loc: u32) -> u32 { return loc * 2u + 1u; }

fn get_rule(a: u32, b: u32) -> u32 {
  let tag_a = get_tag(a);
  let tag_b = get_tag(b);
  if tag_a == VAR || tag_b == VAR { return LINK; }
  if tag_a == REF {
    if tag_b == REF || tag_b == ERA || tag_b == NUM { return VOID; }
    return CALL;
  }
  if tag_b == REF {
    if tag_a == ERA || tag_a == NUM { return VOID; }
    return CALL;
  }
  if tag_a == ERA {
    if tag_b == ERA || tag_b == NUM { return VOID; }
    return ERAS;
  }
  if tag_b == ERA {
    if tag_a == NUM { return VOID; }
    return ERAS;
  }
  if tag_a == NUM {
    if tag_b == NUM { return VOID; }
    if tag_b == OPR { return OPER; }
    if tag_b == SWI { return SWIT; }
    return ERAS;
  }
  if tag_b == NUM {
    if tag_a == OPR { return OPER; }
    if tag_a == SWI { return SWIT; }
    return ERAS;
  }
  if tag_a == tag_b { return ANNI; }
  return COMM;
}

fn should_swap(a: u32, b: u32) -> bool { return get_tag(b) < get_tag(a); }
fn is_high_priority(rule: u32) -> bool { return ((0x1Du >> rule) & 1u) != 0u; }

fn new_u24(val: u32) -> u32 { return (val << 5u) | TY_U24; }
fn get_u24(word: u32) -> u32 { return word >> 5u; }
fn new_i24(val: i32) -> u32 { return (u32(val) << 5u) | TY_I24; }
fn get_i24(word: u32) -> i32 { return (i32(word) << 3) >> 8; }
fn get_typ(word: u32) -> u32 { return word & 0x1Fu; }
fn get_sym(word: u32) -> u32 { return word >> 5u; }
fn new_sym(val: u32) -> u32 { return (val << 5u) | TY_SYM; }

fn operate_u24(op: u32, av: u32, bv: u32) -> u32 {
  switch op {
    case 4u: { return new_u24((av + bv) & 0xFFFFFFu); }
    case 5u: { return new_u24((av - bv) & 0xFFFFFFu); }
    case 6u: { return new_u24((bv - av) & 0xFFFFFFu); }
    case 7u: { return new_u24((av * bv) & 0xFFFFFFu); }
    case 8u: { if bv == 0u { return new_u24(0u); } return new_u24(av / bv); }
    case 9u: { if av == 0u { return new_u24(0u); } return new_u24(bv / av); }
    case 10u: { if bv == 0u { return new_u24(0u); } return new_u24(av % bv); }
    case 11u: { if av == 0u { return new_u24(0u); } return new_u24(bv % av); }
    case 12u: { return new_u24(select(0u, 1u, av == bv)); }
    case 13u: { return new_u24(select(0u, 1u, av != bv)); }
    case 14u: { return new_u24(select(0u, 1u, av < bv)); }
    case 15u: { return new_u24(select(0u, 1u, av > bv)); }
    case 16u: { return new_u24(av & bv); }
    case 17u: { return new_u24(av | bv); }
    case 18u: { return new_u24(av ^ bv); }
    case 19u: { return new_u24((av << (bv & 31u)) & 0xFFFFFFu); }
    case 20u: { return new_u24((bv << (av & 31u)) & 0xFFFFFFu); }
    case 21u: { return new_u24(av >> (bv & 31u)); }
    case 22u: { return new_u24(bv >> (av & 31u)); }
    default: { return new_u24(0u); }
  }
}

fn operate(a: u32, b: u32) -> u32 {
  let at = get_typ(a);
  let bt = get_typ(b);
  if at == TY_SYM && bt == TY_SYM { return new_u24(0u); }
  if at == TY_SYM && bt != TY_SYM { return (b & ~0x1Fu) | get_sym(a); }
  if at != TY_SYM && bt == TY_SYM { return (a & ~0x1Fu) | get_sym(b); }
  if at >= OP_ADD && bt >= OP_ADD { return new_u24(0u); }
  if at < OP_ADD && bt < OP_ADD { return new_u24(0u); }
  var op: u32; var aval: u32; var ty: u32; var bval: u32;
  if at >= OP_ADD { op = at; aval = a; ty = bt; bval = b; }
  else { op = bt; aval = b; ty = at; bval = a; }
  if ty == TY_U24 { return operate_u24(op, get_u24(aval), get_u24(bval)); }
  return new_u24(0u);
}

fn node_load_fst(loc: u32) -> u32 { return atomicLoad(&node_buf[get_fst_idx(loc)]); }
fn node_load_snd(loc: u32) -> u32 { return atomicLoad(&node_buf[get_snd_idx(loc)]); }
fn node_store(loc: u32, fst: u32, snd: u32) {
  atomicStore(&node_buf[get_fst_idx(loc)], fst);
  atomicStore(&node_buf[get_snd_idx(loc)], snd);
}
fn node_take(loc: u32) -> vec2<u32> {
  let fst = atomicExchange(&node_buf[get_fst_idx(loc)], 0u);
  let snd = atomicExchange(&node_buf[get_snd_idx(loc)], 0u);
  return vec2<u32>(fst, snd);
}
fn vars_load(var_idx: u32) -> u32 { return atomicLoad(&vars_buf[var_idx]); }
fn vars_store(var_idx: u32, val: u32) { atomicStore(&vars_buf[var_idx], val); }
fn vars_exchange(var_idx: u32, val: u32) -> u32 { return atomicExchange(&vars_buf[var_idx], val); }
fn vars_take(var_idx: u32) -> u32 { return atomicExchange(&vars_buf[var_idx], 0u); }
fn is_node_free(loc: u32) -> bool { return node_load_fst(loc) == 0u && node_load_snd(loc) == 0u; }
fn is_vars_free(var_idx: u32) -> bool { return vars_load(var_idx) == 0u; }

struct DefHeader { safe: u32, rbag_len: u32, node_len: u32, vars_len: u32, root: u32, rbag_offset: u32, node_offset: u32 }

fn read_def_header(fid: u32) -> DefHeader {
  var offset = 1u;
  for (var i = 0u; i < fid; i++) {
    offset += 1u; offset += 64u;
    let safe = book_buf[offset]; offset += 1u;
    let rbag_len = book_buf[offset]; offset += 1u;
    let node_len = book_buf[offset]; offset += 1u;
    offset += 1u; offset += 1u;
    offset += rbag_len * 2u; offset += node_len * 2u;
  }
  offset += 1u; offset += 64u;
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

var<private> nput: u32;
var<private> vput: u32;
var<private> nloc: array<u32, 256>;
var<private> vloc: array<u32, 256>;
var<private> pri_hi: array<vec2<u32>, 256>;
var<private> pri_lo: array<vec2<u32>, 256>;
var<private> hi_len: u32;
var<private> lo_len: u32;

fn node_alloc(num: u32, gid: u32) -> u32 {
  var got = 0u;
  let node_len = config.node_len;
  let start = (gid * 4096u) % node_len;
  for (var i = 0u; i < node_len && got < num; i++) {
    nput = (start + nput + 1u) % node_len;
    if nput > 0u && is_node_free(nput) { nloc[got] = nput; got++; }
  }
  return got;
}

fn vars_alloc(num: u32, gid: u32) -> u32 {
  var got = 0u;
  let vars_len = config.vars_len;
  let start = (gid * 4096u) % vars_len;
  for (var i = 0u; i < vars_len && got < num; i++) {
    vput = (start + vput + 1u) % vars_len;
    if vput > 0u && is_vars_free(vput) { vloc[got] = vput; got++; }
  }
  return got;
}

fn adjust_port(port: u32) -> u32 {
  let tag = get_tag(port);
  let val = get_val(port);
  if is_nod(port) { return new_port(tag, nloc[val]); }
  if is_var(port) { return new_port(tag, vloc[val]); }
  return port;
}

fn push_redex(a: u32, b: u32) {
  let rule = get_rule(a, b);
  if is_high_priority(rule) { if hi_len < 256u { pri_hi[hi_len] = vec2<u32>(a, b); hi_len++; } }
  else { if lo_len < 256u { pri_lo[lo_len] = vec2<u32>(a, b); lo_len++; } }
}

fn pop_redex() -> vec2<u32> {
  if hi_len > 0u { hi_len--; return pri_hi[hi_len]; }
  if lo_len > 0u { lo_len--; return pri_lo[lo_len]; }
  return vec2<u32>(NONE, NONE);
}

fn has_redex() -> bool { return hi_len > 0u || lo_len > 0u; }

fn link(a_in: u32, b_in: u32) {
  var a = a_in; var b = b_in;
  loop {
    if get_tag(a) != VAR && get_tag(b) == VAR { let tmp = a; a = b; b = tmp; }
    if get_tag(a) != VAR { push_redex(a, b); break; }
    let a_old = vars_exchange(get_val(a), b);
    if a_old == NONE { break; }
    vars_take(get_val(a));
    a = a_old;
  }
}

fn interact_link(a: u32, b: u32) -> bool { link(a, b); return true; }

fn interact_call(a: u32, b: u32, gid: u32) -> bool {
  let fid = get_val(a) & 0xFFFFFFFu;
  let header = read_def_header(fid);
  if get_tag(b) == DUP && header.safe != 0u { return interact_eras(a, b, gid); }
  let got_nodes = node_alloc(header.node_len, gid);
  let got_vars = vars_alloc(header.vars_len, gid);
  if got_nodes < header.node_len || got_vars < header.vars_len { return false; }
  for (var i = 0u; i < header.vars_len; i++) { vars_store(vloc[i], NONE); }
  for (var i = 0u; i < header.node_len; i++) {
    let pair = read_def_node(header, i);
    node_store(nloc[i], adjust_port(pair.x), adjust_port(pair.y));
  }
  for (var i = 0u; i < header.rbag_len; i++) {
    let pair = read_def_rbag(header, i);
    link(adjust_port(pair.x), adjust_port(pair.y));
  }
  link(adjust_port(header.root), b);
  return true;
}

fn interact_void(a: u32, b: u32) -> bool { return true; }

fn interact_eras(a: u32, b: u32, gid: u32) -> bool {
  let b_loc = get_val(b);
  if is_node_free(b_loc) { return false; }
  let b_pair = node_take(b_loc);
  link(a, b_pair.x); link(a, b_pair.y);
  return true;
}

fn interact_anni(a: u32, b: u32, gid: u32) -> bool {
  let a_loc = get_val(a); let b_loc = get_val(b);
  if is_node_free(a_loc) || is_node_free(b_loc) { return false; }
  let a_pair = node_take(a_loc); let b_pair = node_take(b_loc);
  link(a_pair.x, b_pair.x); link(a_pair.y, b_pair.y);
  return true;
}

fn interact_comm(a: u32, b: u32, gid: u32) -> bool {
  let got_nodes = node_alloc(4u, gid);
  let got_vars = vars_alloc(4u, gid);
  if got_nodes < 4u || got_vars < 4u { return false; }
  let a_loc = get_val(a); let b_loc = get_val(b);
  if is_node_free(a_loc) || is_node_free(b_loc) { return false; }
  let a_pair = node_take(a_loc); let b_pair = node_take(b_loc);
  vars_store(vloc[0], NONE); vars_store(vloc[1], NONE);
  vars_store(vloc[2], NONE); vars_store(vloc[3], NONE);
  node_store(nloc[0], new_port(VAR, vloc[0]), new_port(VAR, vloc[1]));
  node_store(nloc[1], new_port(VAR, vloc[2]), new_port(VAR, vloc[3]));
  node_store(nloc[2], new_port(VAR, vloc[0]), new_port(VAR, vloc[2]));
  node_store(nloc[3], new_port(VAR, vloc[1]), new_port(VAR, vloc[3]));
  link(new_port(get_tag(b), nloc[0]), a_pair.x);
  link(new_port(get_tag(b), nloc[1]), a_pair.y);
  link(new_port(get_tag(a), nloc[2]), b_pair.x);
  link(new_port(get_tag(a), nloc[3]), b_pair.y);
  return true;
}

fn interact_oper(a: u32, b: u32, gid: u32) -> bool {
  let b_loc = get_val(b);
  if is_node_free(b_loc) { return false; }
  let av = get_val(a);
  let b_pair = node_take(b_loc);
  var b1 = b_pair.x;
  let b2 = b_pair.y;
  if get_tag(b1) == NUM {
    let bv = get_val(b1);
    let cv = operate(av, bv);
    link(new_port(NUM, cv), b2);
  } else {
    let got_nodes = node_alloc(1u, gid);
    if got_nodes < 1u { return false; }
    node_store(nloc[0], new_port(NUM, av), b2);
    link(b1, new_port(OPR, nloc[0]));
  }
  return true;
}

fn interact_swit(a: u32, b: u32, gid: u32) -> bool {
  let b_loc = get_val(b);
  if is_node_free(b_loc) { return false; }
  let got_nodes = node_alloc(2u, gid);
  if got_nodes < 2u { return false; }
  let av = get_u24(get_val(a));
  let b_pair = node_take(b_loc);
  let b1 = b_pair.x; let b2 = b_pair.y;
  if av == 0u {
    node_store(nloc[0], b2, new_port(ERA, 0u));
    link(new_port(CON, nloc[0]), b1);
  } else {
    node_store(nloc[0], new_port(ERA, 0u), new_port(CON, nloc[1]));
    node_store(nloc[1], new_port(NUM, new_u24(av - 1u)), b2);
    link(new_port(CON, nloc[0]), b1);
  }
  return true;
}

fn interact(a_in: u32, b_in: u32, gid: u32) -> bool {
  var a = a_in; var b = b_in;
  var rule = get_rule(a, b);
  if get_tag(a) == REF && b == ROOT { rule = CALL; }
  else if should_swap(a, b) { let tmp = a; a = b; b = tmp; }
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let gid = global_id.x;
  hi_len = 0u; lo_len = 0u;
  nput = gid * 128u; vput = gid * 128u;
  let rbag_start = gid * RLEN;
  let my_len = atomicLoad(&rbag_len[gid]);
  for (var i = 0u; i < my_len && i < RLEN; i++) {
    let idx = (rbag_start + i) * 2u;
    push_redex(rbag_buf[idx], rbag_buf[idx + 1u]);
  }
  atomicStore(&rbag_len[gid], 0u);
  var itrs = 0u;
  let max_itrs = 65536u;
  while has_redex() && itrs < max_itrs {
    let redex = pop_redex();
    if redex.x == NONE { break; }
    if interact(redex.x, redex.y, gid) { itrs++; }
    else { push_redex(redex.x, redex.y); break; }
  }
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
  atomicAdd(&config.itrs, itrs);
  if store_idx == 0u { atomicAdd(&config.done, 1u); }
}

@compute @workgroup_size(1)
fn boot(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let defs_len = book_buf[0];
  var main_fid = 0u;
  var offset = 1u;
  for (var i = 0u; i < defs_len; i++) {
    offset += 1u;
    let c0 = book_buf[offset] & 0xFFu;
    let c1 = (book_buf[offset] >> 8u) & 0xFFu;
    let c2 = (book_buf[offset] >> 16u) & 0xFFu;
    let c3 = (book_buf[offset] >> 24u) & 0xFFu;
    if c0 == 109u && c1 == 97u && c2 == 105u && c3 == 110u { main_fid = i; }
    offset += 64u; offset += 1u;
    let rbag_len_val = book_buf[offset]; offset += 1u;
    let node_len_val = book_buf[offset]; offset += 1u;
    offset += 1u; offset += 1u;
    offset += rbag_len_val * 2u;
    offset += node_len_val * 2u;
  }
  vars_store(0x1FFFFFFFu, NONE);
  rbag_buf[0] = new_port(REF, main_fid);
  rbag_buf[1] = ROOT;
  atomicStore(&rbag_len[0], 1u);
  atomicStore(&config.done, 0u);
  atomicStore(&config.itrs, 0u);
}
`;

/**
 * HVM WebGPU Runtime Class
 */
export class HVMRuntime {
  constructor() {
    this.device = null;
    this.queue = null;
    this.buffers = {};
    this.pipelines = {};
    this.bindGroup = null;
    this.initialized = false;
  }

  /**
   * Check if WebGPU is supported
   */
  static isSupported() {
    return typeof navigator !== 'undefined' && !!navigator.gpu;
  }

  /**
   * Initialize the WebGPU runtime
   */
  async init() {
    if (!HVMRuntime.isSupported()) {
      throw new Error('WebGPU is not supported in this browser');
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });

    if (!adapter) {
      throw new Error('Failed to get GPU adapter');
    }

    this.device = await adapter.requestDevice();
    this.queue = this.device.queue;
    this.adapterInfo = adapter.info;
    this.initialized = true;

    return this;
  }

  /**
   * Get adapter information
   */
  getAdapterInfo() {
    return this.adapterInfo || { vendor: 'unknown', architecture: 'unknown' };
  }

  /**
   * Run an HVM program from a book buffer
   * @param {Uint8Array} bookData - Serialized book buffer
   * @param {Object} config - Runtime configuration
   */
  async run(bookData, config = {}) {
    if (!this.initialized) {
      await this.init();
    }

    const {
      workgroups = 64,
      nodeLen = 1 << 18,
      varsLen = 1 << 18,
      maxIterations = 1000
    } = config;

    // Ensure book data is 4-byte aligned
    const alignedBookData = new Uint8Array(Math.ceil(bookData.length / 4) * 4);
    alignedBookData.set(bookData);

    // Create buffers
    const nodeBufSize = nodeLen * 2 * 4;
    const varsBufSize = varsLen * 4;
    const rbagSize = workgroups * 256 * 256 * 2 * 4;
    const rbagLenSize = workgroups * 256 * 4;

    this.buffers.node = this.device.createBuffer({
      size: nodeBufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.buffers.vars = this.device.createBuffer({
      size: varsBufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.buffers.rbag = this.device.createBuffer({
      size: rbagSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.buffers.rbagLen = this.device.createBuffer({
      size: rbagLenSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.buffers.book = this.device.createBuffer({
      size: alignedBookData.length,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    new Uint8Array(this.buffers.book.getMappedRange()).set(alignedBookData);
    this.buffers.book.unmap();

    const configData = new Uint32Array([
      nodeLen,
      varsLen,
      workgroups * 256,
      0,
      0,
      0,
      0,
      0
    ]);

    this.buffers.config = this.device.createBuffer({
      size: configData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint32Array(this.buffers.config.getMappedRange()).set(configData);
    this.buffers.config.unmap();

    this.buffers.staging = this.device.createBuffer({
      size: configData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    this.buffers.resultStaging = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // Create shader module
    const shaderModule = this.device.createShaderModule({
      code: HVM_SHADER
    });

    // Create bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
      ]
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.buffers.node } },
        { binding: 1, resource: { buffer: this.buffers.vars } },
        { binding: 2, resource: { buffer: this.buffers.rbag } },
        { binding: 3, resource: { buffer: this.buffers.rbagLen } },
        { binding: 4, resource: { buffer: this.buffers.book } },
        { binding: 5, resource: { buffer: this.buffers.config } }
      ]
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });

    this.pipelines.boot = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'boot' }
    });

    this.pipelines.main = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'main' }
    });

    const startTime = performance.now();

    // Run boot kernel
    {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.boot);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      this.queue.submit([encoder.finish()]);
    }

    // Main evaluation loop
    let totalItrs = 0;
    let iterations = 0;

    while (iterations < maxIterations) {
      iterations++;

      // Reset done counter
      this.queue.writeBuffer(this.buffers.config, 20, new Uint32Array([0]));

      // Run main kernel
      {
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.main);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(this.buffers.config, 0, this.buffers.staging, 0, 32);
        this.queue.submit([encoder.finish()]);
      }

      // Read results
      await this.buffers.staging.mapAsync(GPUMapMode.READ);
      const data = new Uint32Array(this.buffers.staging.getMappedRange());
      const done = data[5];
      const itrs = data[6];
      this.buffers.staging.unmap();

      totalItrs += itrs;

      if (done >= workgroups * 256) {
        break;
      }

      // Reset iteration counter
      this.queue.writeBuffer(this.buffers.config, 24, new Uint32Array([0]));
    }

    const elapsed = (performance.now() - startTime) / 1000;

    // Read result from root variable
    {
      const encoder = this.device.createCommandEncoder();
      const rootIdx = 0x1FFFFFFF;
      // For simplicity, read from the beginning of vars buffer where result typically ends up
      encoder.copyBufferToBuffer(this.buffers.vars, 0, this.buffers.resultStaging, 0, 4);
      this.queue.submit([encoder.finish()]);
    }

    await this.buffers.resultStaging.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(this.buffers.resultStaging.getMappedRange());
    const resultPort = resultData[0];
    this.buffers.resultStaging.unmap();

    // Cleanup buffers
    for (const buf of Object.values(this.buffers)) {
      buf.destroy();
    }
    this.buffers = {};

    return {
      interactions: totalItrs,
      time: elapsed,
      mips: totalItrs / elapsed / 1e6,
      result: resultPort,
      iterations
    };
  }

  /**
   * Destroy the runtime and release resources
   */
  destroy() {
    for (const buf of Object.values(this.buffers)) {
      buf.destroy();
    }
    this.buffers = {};
    this.device = null;
    this.initialized = false;
  }
}

/**
 * Simple HVM AST Parser (subset)
 * Parses HVM IR to a book structure
 */
export class HVMParser {
  constructor(code) {
    this.code = code;
    this.pos = 0;
  }

  static parse(code) {
    const parser = new HVMParser(code);
    return parser.parseBook();
  }

  parseBook() {
    const defs = [];
    while (this.pos < this.code.length) {
      this.skipWhitespace();
      if (this.pos >= this.code.length) break;
      if (this.peek() === '@') {
        defs.push(this.parseDef());
      } else {
        this.pos++;
      }
    }
    return { defs };
  }

  parseDef() {
    this.expect('@');
    const name = this.parseName();
    this.skipWhitespace();
    this.expect('=');
    this.skipWhitespace();

    const { root, nodes, vars, redexes } = this.parseNet();

    return {
      name,
      safe: true,
      root,
      nodes,
      vars,
      redexes
    };
  }

  parseNet() {
    const nodes = [];
    const redexes = [];
    let vars = 0;

    const root = this.parsePort(nodes, { vars: 0 });
    vars = Math.max(vars, root.maxVar || 0);

    // Parse redexes (& ... ~ ...)
    while (true) {
      this.skipWhitespace();
      if (this.peek() !== '&') break;
      this.pos++; // skip &

      const strict = this.peek() === '!';
      if (strict) this.pos++;

      this.skipWhitespace();
      const a = this.parsePort(nodes, { vars });
      vars = Math.max(vars, a.maxVar || 0);

      this.skipWhitespace();
      this.expect('~');
      this.skipWhitespace();

      const b = this.parsePort(nodes, { vars });
      vars = Math.max(vars, b.maxVar || 0);

      redexes.push({ a: a.port, b: b.port });
    }

    return { root: root.port, nodes, vars: vars + 1, redexes };
  }

  parsePort(nodes, ctx) {
    this.skipWhitespace();
    const c = this.peek();

    if (c === '(') {
      // Constructor/Pair: (a b)
      this.pos++;
      this.skipWhitespace();
      const fst = this.parsePort(nodes, ctx);
      ctx.vars = Math.max(ctx.vars, fst.maxVar || 0);
      this.skipWhitespace();
      const snd = this.parsePort(nodes, ctx);
      ctx.vars = Math.max(ctx.vars, snd.maxVar || 0);
      this.skipWhitespace();
      this.expect(')');

      const nodeIdx = nodes.length;
      nodes.push({ fst: fst.port, snd: snd.port });
      return { port: this.makePort(4, nodeIdx), maxVar: Math.max(fst.maxVar || 0, snd.maxVar || 0) }; // CON
    }

    if (c === '{') {
      // Duplicator: {a b}
      this.pos++;
      this.skipWhitespace();
      const fst = this.parsePort(nodes, ctx);
      ctx.vars = Math.max(ctx.vars, fst.maxVar || 0);
      this.skipWhitespace();
      const snd = this.parsePort(nodes, ctx);
      ctx.vars = Math.max(ctx.vars, snd.maxVar || 0);
      this.skipWhitespace();
      this.expect('}');

      const nodeIdx = nodes.length;
      nodes.push({ fst: fst.port, snd: snd.port });
      return { port: this.makePort(5, nodeIdx), maxVar: Math.max(fst.maxVar || 0, snd.maxVar || 0) }; // DUP
    }

    if (c === '@') {
      // Reference
      this.pos++;
      const name = this.parseName();
      return { port: { type: 'ref', name }, maxVar: 0 };
    }

    if (c === '*') {
      // Eraser
      this.pos++;
      return { port: this.makePort(2, 0), maxVar: 0 }; // ERA
    }

    if (c === '?') {
      // Switch: ?(... ...)
      this.pos++;
      this.skipWhitespace();
      this.expect('(');
      this.skipWhitespace();
      const fst = this.parsePort(nodes, ctx);
      ctx.vars = Math.max(ctx.vars, fst.maxVar || 0);
      this.skipWhitespace();
      const snd = this.parsePort(nodes, ctx);
      ctx.vars = Math.max(ctx.vars, snd.maxVar || 0);
      this.skipWhitespace();
      this.expect(')');

      const nodeIdx = nodes.length;
      nodes.push({ fst: fst.port, snd: snd.port });
      return { port: this.makePort(7, nodeIdx), maxVar: Math.max(fst.maxVar || 0, snd.maxVar || 0) }; // SWI
    }

    if (c === '$') {
      // Operator: $([op] port) or $(port port)
      this.pos++;
      this.skipWhitespace();
      this.expect('(');
      this.skipWhitespace();
      const fst = this.parsePort(nodes, ctx);
      ctx.vars = Math.max(ctx.vars, fst.maxVar || 0);
      this.skipWhitespace();
      const snd = this.parsePort(nodes, ctx);
      ctx.vars = Math.max(ctx.vars, snd.maxVar || 0);
      this.skipWhitespace();
      this.expect(')');

      const nodeIdx = nodes.length;
      nodes.push({ fst: fst.port, snd: snd.port });
      return { port: this.makePort(6, nodeIdx), maxVar: Math.max(fst.maxVar || 0, snd.maxVar || 0) }; // OPR
    }

    if (c === '[') {
      // Operator symbol: [+], [*2], etc.
      this.pos++;
      let op = '';
      while (this.peek() !== ']') {
        op += this.code[this.pos++];
      }
      this.expect(']');
      const opVal = this.parseOpSymbol(op);
      return { port: this.makePort(3, opVal), maxVar: 0 }; // NUM with op symbol
    }

    if (/[0-9]/.test(c) || (c === '-' && /[0-9]/.test(this.code[this.pos + 1]))) {
      // Number
      const num = this.parseNumber();
      return { port: this.makePort(3, (num << 5) | 1), maxVar: 0 }; // NUM (u24)
    }

    if (/[a-z]/.test(c)) {
      // Variable
      const name = this.parseName();
      const varIdx = ctx.vars++;
      return { port: this.makePort(0, varIdx), maxVar: varIdx }; // VAR
    }

    throw new Error(`Unexpected character: ${c} at position ${this.pos}`);
  }

  makePort(tag, val) {
    return (val << 3) | tag;
  }

  parseOpSymbol(op) {
    const ops = {
      '+': 4, '-': 5, '*': 7, '/': 8, '%': 10,
      '==': 12, '!=': 13, '<': 14, '>': 15,
      '&': 16, '|': 17, '^': 18, '<<': 19, '>>': 21
    };

    // Check for operations with immediate values like [*2], [+1]
    for (const [sym, code] of Object.entries(ops)) {
      if (op.startsWith(sym)) {
        const rest = op.slice(sym.length);
        if (rest) {
          const num = parseInt(rest);
          return (num << 5) | code;
        }
        return (0 << 5) | code;
      }
    }

    return (0 << 5) | 0; // TY_SYM
  }

  parseNumber() {
    let num = '';
    if (this.peek() === '-') {
      num += this.code[this.pos++];
    }
    while (/[0-9]/.test(this.peek())) {
      num += this.code[this.pos++];
    }
    return parseInt(num);
  }

  parseName() {
    let name = '';
    while (/[a-zA-Z0-9_]/.test(this.peek())) {
      name += this.code[this.pos++];
    }
    return name;
  }

  peek() {
    return this.code[this.pos];
  }

  expect(c) {
    if (this.code[this.pos] !== c) {
      throw new Error(`Expected '${c}' but got '${this.code[this.pos]}' at position ${this.pos}`);
    }
    this.pos++;
  }

  skipWhitespace() {
    while (this.pos < this.code.length && /[\s\n\r]/.test(this.code[this.pos])) {
      this.pos++;
    }
  }
}

/**
 * Book Serializer - converts parsed AST to binary format
 */
export class BookSerializer {
  static serialize(book) {
    const buffer = [];

    // Resolve references to function IDs
    const nameToId = {};
    book.defs.forEach((def, idx) => {
      nameToId[def.name] = idx;
    });

    // Write defs_len
    buffer.push(book.defs.length);

    for (let i = 0; i < book.defs.length; i++) {
      const def = book.defs[i];

      // Write fid
      buffer.push(i);

      // Write name (256 bytes = 64 u32s)
      const nameBytes = new Uint8Array(256);
      for (let j = 0; j < def.name.length && j < 255; j++) {
        nameBytes[j] = def.name.charCodeAt(j);
      }
      for (let j = 0; j < 64; j++) {
        buffer.push(
          nameBytes[j * 4] |
          (nameBytes[j * 4 + 1] << 8) |
          (nameBytes[j * 4 + 2] << 16) |
          (nameBytes[j * 4 + 3] << 24)
        );
      }

      // Write safe
      buffer.push(def.safe ? 1 : 0);

      // Write rbag_len
      buffer.push(def.redexes.length);

      // Write node_len
      buffer.push(def.nodes.length);

      // Write vars_len
      buffer.push(def.vars);

      // Resolve and write root
      const resolvedRoot = this.resolvePort(def.root, nameToId);
      buffer.push(resolvedRoot);

      // Write rbag_buf
      for (const redex of def.redexes) {
        buffer.push(this.resolvePort(redex.a, nameToId));
        buffer.push(this.resolvePort(redex.b, nameToId));
      }

      // Write node_buf
      for (const node of def.nodes) {
        buffer.push(this.resolvePort(node.fst, nameToId));
        buffer.push(this.resolvePort(node.snd, nameToId));
      }
    }

    return new Uint8Array(new Uint32Array(buffer).buffer);
  }

  static resolvePort(port, nameToId) {
    if (typeof port === 'number') {
      return port;
    }
    if (port && port.type === 'ref') {
      const fid = nameToId[port.name];
      if (fid === undefined) {
        throw new Error(`Unknown reference: @${port.name}`);
      }
      return (fid << 3) | 1; // REF tag
    }
    return port;
  }
}

/**
 * Compile HVM code to a book buffer
 */
export function compileHVM(code) {
  const book = HVMParser.parse(code);
  return BookSerializer.serialize(book);
}

/**
 * Run HVM code on WebGPU
 */
export async function runHVM(code, config = {}) {
  const bookData = compileHVM(code);
  const runtime = new HVMRuntime();
  await runtime.init();
  const result = await runtime.run(bookData, config);
  runtime.destroy();
  return result;
}

// Export for use in tests and browser
export default {
  HVMRuntime,
  HVMParser,
  BookSerializer,
  compileHVM,
  runHVM,
  HVM_SHADER
};
