type token =
  | EOF
  | TDot
  | TComma
  | TSemicolon
  | TPtr
  | TEqual
  | TLParen
  | TRParen
  | TLBrace
  | TRBrace
  | TLBrack
  | TRBrack
  | Tpublic
  | Tinclude
  | Tconst
  | Tidentifier of (string)
  | Tnumber of (int)
  | Tstring of (string)
  | Tchar
  | Tshort
  | Tunsigned
  | Tint
  | Tfloat
  | Tdouble
  | Tint8
  | Tint16
  | Tint32
  | Tint64
  | Tuint8
  | Tuint16
  | Tuint32
  | Tuint64
  | Tsizet
  | Twchar
  | Tvoid
  | Tlong
  | Tstruct
  | Tunion
  | Tenum
  | TCudaStream
  | Tenclave
  | Tfrom
  | Timport
  | Ttrusted
  | Tuntrusted
  | Tallow
  | Tpropagate_errno

open Parsing;;
let _ = parse_error;;
# 33 "Parser.mly"
open Util				(* for failwithf *)

(* Here we defined some helper routines to check attributes.
 *
 * An alternative approach is to code these rules in Lexer/Parser but
 * it has several drawbacks:
 *
 * 1. Bad extensibility;
 * 2. It grows the table size and down-graded the parsing time;
 * 3. It makes error reporting rigid this way.
 *)

let get_string_from_attr (v: Ast.attr_value) (err_func: int -> string) =
  match v with
      Ast.AString s -> s
    | Ast.ANumber n -> err_func n

(* Check whether 'size' or 'sizefunc' is specified. *)
let has_size (sattr: Ast.ptr_size) =
  sattr.Ast.ps_size <> None || sattr.Ast.ps_sizefunc <> None

(* Pointers can have the following attributes:
 *
 * 'size'     - specifies the size of the pointer.
 *              e.g. size = 4, size = val ('val' is a parameter);
 *
 * 'count'    - indicates how many of items is managed by the pointer
 *              e.g. count = 100, count = n ('n' is a parameter);
 *
 * 'sizefunc' - use a function to compute the size of the pointer.
 *              e.g. sizefunc = get_ptr_size
 *
 * 'string'   - indicate the pointer is managing a C string;
 * 'wstring'  - indicate the pointer is managing a wide char string.
 *
 * 'isptr'    - to specify that the foreign type is a pointer.
 * 'isary'    - to specify that the foreign type is an array.
 * 'readonly' - to specify that the foreign type has a 'const' qualifier.
 *
 * 'user_check' - inhibit Edger8r from generating code to check the pointer.
 *
 * 'in'       - the pointer is used as input
 * 'out'      - the pointer is used as output
 *
 * 'offset'   - the pointer should be added the offset caused by the chkpt-restore
 *
 * Note that 'size' and 'sizefunc' are mutual exclusive (but they can
 * be used together with 'count'.  'string' and 'wstring' indicates 'isptr',
 * and they cannot use with only an 'out' attribute.
 *)
let get_ptr_attr (attr_list: (string * Ast.attr_value) list) =
  let get_new_dir (cds: string) (cda: Ast.ptr_direction) (old: Ast.ptr_direction) =
    if old = Ast.PtrNoDirection then cda
    else if old = Ast.PtrInOut  then failwithf "duplicated attribute: `%s'" cds
    else if old = cda           then failwithf "duplicated attribute: `%s'" cds
    else Ast.PtrInOut
  in
  let update_attr (key: string) (value: Ast.attr_value) (res: Ast.ptr_attr) =
    match key with
        "size"     ->
        { res with Ast.pa_size = { res.Ast.pa_size with Ast.ps_size  = Some value }}
      | "count"    ->
        { res with Ast.pa_size = { res.Ast.pa_size with Ast.ps_count = Some value }}
      | "sizefunc" ->
        let efn n = failwithf "invalid function name (%d) for `sizefunc'" n in
        let funcname = get_string_from_attr value efn
        in { res with Ast.pa_size =
            { res.Ast.pa_size with Ast.ps_sizefunc = Some funcname }}
      | "string"  -> { res with Ast.pa_isptr = true; Ast.pa_isstr = true; }
      | "wstring" -> { res with Ast.pa_isptr = true; Ast.pa_iswstr = true; }
      | "isptr"   -> { res with Ast.pa_isptr = true }
      | "isary"   -> { res with Ast.pa_isary = true }

      | "readonly" -> { res with Ast.pa_rdonly = true }
      | "user_check" -> { res with Ast.pa_chkptr = false }

      | "in"  ->
        let newdir = get_new_dir "in"  Ast.PtrIn  res.Ast.pa_direction
        in { res with Ast.pa_direction = newdir }
      | "out" ->
        let newdir = get_new_dir "out" Ast.PtrOut res.Ast.pa_direction
        in { res with Ast.pa_direction = newdir }
      (* alex modified on 12 Jan 2023 *)
      | "offset" -> { res with Ast.pa_offset = true; }

      | _ -> failwithf "unknown attribute: %s" key
  in
  let rec do_get_ptr_attr alist res_attr =
    match alist with
        [] -> res_attr
      | (k,v) :: xs -> do_get_ptr_attr xs (update_attr k v res_attr)
  in
  let has_str_attr (pattr: Ast.ptr_attr) =
    if pattr.Ast.pa_isstr && pattr.Ast.pa_iswstr
    then failwith "`string' and `wstring' are mutual exclusive"
    else (pattr.Ast.pa_isstr || pattr.Ast.pa_iswstr)
  in
  let check_invalid_ptr_size (pattr: Ast.ptr_attr) =
    let ps = pattr.Ast.pa_size in
      if ps.Ast.ps_size <> None && ps.Ast.ps_sizefunc <> None
      then failwith  "`size' and `sizefunc' cannot be used at the same time"
      else
        if ps <> Ast.empty_ptr_size && has_str_attr pattr
        then failwith "size attributes are mutual exclusive with (w)string attribute"
        else
          if (ps <> Ast.empty_ptr_size || has_str_attr pattr) &&
            pattr.Ast.pa_direction = Ast.PtrNoDirection
          then failwith "size/string attributes must be used with pointer direction"
          else pattr
  in
  let check_ptr_dir (pattr: Ast.ptr_attr) =
    if pattr.Ast.pa_direction <> Ast.PtrNoDirection && pattr.Ast.pa_chkptr = false
    then failwith "pointer direction and `user_check' are mutual exclusive"
    else
      if pattr.Ast.pa_direction = Ast.PtrNoDirection && pattr.Ast.pa_chkptr
      then failwith "pointer/array should have direction attribute or `user_check'"
      else
        if pattr.Ast.pa_direction = Ast.PtrOut && (has_str_attr pattr || pattr.Ast.pa_size.Ast.ps_sizefunc <> None)
        then failwith "string/wstring/sizefunc should be used with an `in' attribute"
        else pattr
  in
  let check_invalid_ary_attr (pattr: Ast.ptr_attr) =
    if pattr.Ast.pa_size <> Ast.empty_ptr_size
    then failwith "Pointer size attributes cannot be used with foreign array"
    else
      if not pattr.Ast.pa_isptr
      then
        (* 'pa_chkptr' is default to true unless user specifies 'user_check' *)
        if pattr.Ast.pa_chkptr && pattr.Ast.pa_direction = Ast.PtrNoDirection
        then failwith "array must have direction attribute or `user_check'"
        else pattr
      else
        if has_str_attr pattr
        then failwith "`isary' cannot be used with `string/wstring' together"
        else failwith "`isary' cannot be used with `isptr' together"
  in
  let pattr = do_get_ptr_attr attr_list { Ast.pa_direction = Ast.PtrNoDirection;
                                          Ast.pa_size = Ast.empty_ptr_size;
                                          Ast.pa_isptr = false;
                                          Ast.pa_isary = false;
                                          Ast.pa_isstr = false;
                                          Ast.pa_iswstr = false;
                                          Ast.pa_rdonly = false;
                                          Ast.pa_chkptr = true;
                                          Ast.pa_offset = false;
                                        }
  in
    if pattr.Ast.pa_isary
    then check_invalid_ary_attr pattr
    else check_invalid_ptr_size pattr |> check_ptr_dir

(* Untrusted functions can have these attributes:
 *
 * a. 3 mutual exclusive calling convention specifier:
 *     'stdcall', 'fastcall', 'cdecl'.
 *
 * b. 'dllimport' - to import a public symbol.
 *)
let get_func_attr (attr_list: (string * Ast.attr_value) list) =
  let get_new_callconv (key: string) (cur: Ast.call_conv) (old: Ast.call_conv) =
    if old <> Ast.CC_NONE then
      failwithf "unexpected `%s',  conflict with `%s'." key (Ast.get_call_conv_str old)
    else cur
  in
  let update_attr (key: string) (value: Ast.attr_value) (res: Ast.func_attr) =
    match key with
    | "stdcall"  ->
      let callconv = get_new_callconv key Ast.CC_STDCALL res.Ast.fa_convention
      in { res with Ast.fa_convention = callconv}
    | "fastcall" ->
      let callconv = get_new_callconv key Ast.CC_FASTCALL res.Ast.fa_convention
      in { res with Ast.fa_convention = callconv}
    | "cdecl"    ->
      let callconv = get_new_callconv key Ast.CC_CDECL res.Ast.fa_convention
      in { res with Ast.fa_convention = callconv}
    | "dllimport" ->
      if res.Ast.fa_dllimport then failwith "duplicated attribute: `dllimport'"
      else { res with Ast.fa_dllimport = true }
    | _ -> failwithf "invalid function attribute: %s" key
  in
  let rec do_get_func_attr alist res_attr =
    match alist with
      [] -> res_attr
    | (k,v) :: xs -> do_get_func_attr xs (update_attr k v res_attr)
  in do_get_func_attr attr_list { Ast.fa_dllimport = false;
                                  Ast.fa_convention= Ast.CC_NONE;
                                }

(* Some syntax checking against pointer attributes.
 * range: (Lexing.position * Lexing.position)
 *)
let check_ptr_attr (fd: Ast.func_decl) range =
  let fname = fd.Ast.fname in
  let check_const (pattr: Ast.ptr_attr) (identifier: string) =
    let raise_err_direction (direction:string) =
      failwithf "`%s': `%s' is readonly - cannot be used with `%s'"
        fname identifier direction
    in
      if pattr.Ast.pa_rdonly
      then
        match pattr.Ast.pa_direction with
            Ast.PtrOut | Ast.PtrInOut -> raise_err_direction "out"
          | _ -> ()
      else ()
  in
  let check_void_ptr_size (pattr: Ast.ptr_attr) (identifier: string) =
    if pattr.Ast.pa_chkptr && (not (has_size pattr.Ast.pa_size) && (not pattr.Ast.pa_isptr))
    then failwithf "`%s': void pointer `%s' - buffer size unknown" fname identifier
    else ()
  in
  let checker (pd: Ast.pdecl) =
    let pt, declr = pd in
    let identifier = declr.Ast.identifier in
      match pt with
          Ast.PTVal _ -> ()
        | Ast.PTPtr(atype, pattr) ->
          if atype <> Ast.Ptr(Ast.Void) then check_const pattr identifier
          else (* 'void' pointer, check there is a size or 'user_check' *)
            check_void_ptr_size pattr identifier
  in
    List.iter checker fd.Ast.plist
# 275 "Parser.ml"
let yytransl_const = [|
    0 (* EOF *);
  257 (* TDot *);
  258 (* TComma *);
  259 (* TSemicolon *);
  260 (* TPtr *);
  261 (* TEqual *);
  262 (* TLParen *);
  263 (* TRParen *);
  264 (* TLBrace *);
  265 (* TRBrace *);
  266 (* TLBrack *);
  267 (* TRBrack *);
  268 (* Tpublic *);
  269 (* Tinclude *);
  270 (* Tconst *);
  274 (* Tchar *);
  275 (* Tshort *);
  276 (* Tunsigned *);
  277 (* Tint *);
  278 (* Tfloat *);
  279 (* Tdouble *);
  280 (* Tint8 *);
  281 (* Tint16 *);
  282 (* Tint32 *);
  283 (* Tint64 *);
  284 (* Tuint8 *);
  285 (* Tuint16 *);
  286 (* Tuint32 *);
  287 (* Tuint64 *);
  288 (* Tsizet *);
  289 (* Twchar *);
  290 (* Tvoid *);
  291 (* Tlong *);
  292 (* Tstruct *);
  293 (* Tunion *);
  294 (* Tenum *);
  295 (* TCudaStream *);
  296 (* Tenclave *);
  297 (* Tfrom *);
  298 (* Timport *);
  299 (* Ttrusted *);
  300 (* Tuntrusted *);
  301 (* Tallow *);
  302 (* Tpropagate_errno *);
    0|]

let yytransl_block = [|
  271 (* Tidentifier *);
  272 (* Tnumber *);
  273 (* Tstring *);
    0|]

let yylhs = "\255\255\
\002\000\002\000\003\000\003\000\004\000\004\000\005\000\005\000\
\006\000\006\000\006\000\006\000\006\000\007\000\007\000\007\000\
\007\000\007\000\007\000\007\000\007\000\007\000\007\000\007\000\
\007\000\007\000\007\000\007\000\007\000\007\000\007\000\007\000\
\007\000\007\000\011\000\011\000\012\000\013\000\014\000\014\000\
\015\000\015\000\015\000\016\000\016\000\017\000\017\000\018\000\
\018\000\018\000\018\000\019\000\019\000\020\000\020\000\021\000\
\021\000\021\000\008\000\009\000\010\000\022\000\024\000\025\000\
\025\000\026\000\026\000\027\000\027\000\028\000\028\000\028\000\
\029\000\029\000\029\000\023\000\023\000\030\000\031\000\031\000\
\032\000\033\000\033\000\034\000\035\000\035\000\036\000\036\000\
\037\000\037\000\038\000\038\000\041\000\041\000\039\000\039\000\
\040\000\040\000\042\000\042\000\044\000\044\000\044\000\045\000\
\045\000\046\000\047\000\047\000\043\000\043\000\048\000\048\000\
\048\000\049\000\049\000\049\000\049\000\049\000\050\000\001\000\
\000\000"

let yylen = "\002\000\
\001\000\002\000\001\000\001\000\002\000\003\000\000\000\001\000\
\002\000\003\000\002\000\001\000\001\000\001\000\001\000\001\000\
\001\000\002\000\001\000\001\000\001\000\001\000\001\000\001\000\
\001\000\001\000\001\000\001\000\001\000\001\000\001\000\001\000\
\001\000\001\000\001\000\002\000\002\000\003\000\001\000\002\000\
\001\000\001\000\002\000\001\000\002\000\001\000\002\000\002\000\
\001\000\004\000\003\000\002\000\003\000\001\000\003\000\003\000\
\003\000\001\000\002\000\002\000\002\000\004\000\004\000\004\000\
\004\000\000\000\001\000\001\000\003\000\001\000\003\000\003\000\
\001\000\001\000\001\000\002\000\003\000\002\000\001\000\003\000\
\001\000\004\000\004\000\002\000\001\000\002\000\005\000\005\000\
\001\000\002\000\001\000\002\000\000\000\001\000\000\000\004\000\
\000\000\003\000\003\000\004\000\002\000\003\000\003\000\001\000\
\003\000\002\000\000\000\001\000\004\000\003\000\000\000\003\000\
\004\000\000\000\002\000\003\000\003\000\002\000\004\000\003\000\
\002\000"

let yydefred = "\000\000\
\000\000\000\000\000\000\121\000\000\000\114\000\000\000\000\000\
\120\000\119\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\073\000\074\000\075\000\000\000\
\000\000\115\000\118\000\084\000\059\000\060\000\000\000\061\000\
\081\000\000\000\000\000\000\000\000\000\000\000\000\000\117\000\
\116\000\000\000\000\000\000\000\068\000\000\000\085\000\000\000\
\000\000\000\000\000\000\000\000\000\000\034\000\001\000\003\000\
\000\000\016\000\017\000\019\000\020\000\021\000\022\000\023\000\
\024\000\025\000\026\000\027\000\028\000\029\000\000\000\000\000\
\030\000\014\000\000\000\012\000\000\000\015\000\000\000\031\000\
\032\000\033\000\000\000\000\000\000\000\000\000\000\000\000\000\
\064\000\000\000\083\000\079\000\000\000\086\000\000\000\000\000\
\094\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\002\000\000\000\008\000\000\000\018\000\005\000\009\000\
\035\000\000\000\000\000\078\000\062\000\000\000\076\000\063\000\
\065\000\071\000\072\000\069\000\000\000\087\000\000\000\088\000\
\052\000\000\000\000\000\054\000\000\000\000\000\000\000\039\000\
\000\000\000\000\000\000\000\000\000\000\098\000\006\000\010\000\
\036\000\047\000\077\000\080\000\096\000\000\000\000\000\053\000\
\037\000\000\000\000\000\099\000\000\000\000\000\040\000\000\000\
\000\000\000\000\108\000\110\000\056\000\057\000\055\000\038\000\
\101\000\000\000\000\000\049\000\000\000\000\000\000\000\104\000\
\100\000\109\000\112\000\000\000\000\000\102\000\106\000\000\000\
\048\000\000\000\103\000\113\000\000\000\000\000\105\000\000\000"

let yydgoto = "\002\000\
\004\000\074\000\075\000\076\000\077\000\078\000\079\000\080\000\
\081\000\082\000\114\000\135\000\136\000\137\000\138\000\083\000\
\116\000\173\000\103\000\131\000\132\000\021\000\084\000\022\000\
\023\000\043\000\044\000\045\000\024\000\085\000\093\000\034\000\
\025\000\047\000\048\000\027\000\049\000\052\000\050\000\053\000\
\098\000\104\000\105\000\156\000\175\000\176\000\164\000\141\000\
\008\000\005\000"

let yysindex = "\011\000\
\234\254\000\000\118\255\000\000\088\255\000\000\123\000\249\254\
\000\000\000\000\174\255\149\255\204\255\080\255\205\255\215\255\
\245\255\017\000\018\000\019\000\000\000\000\000\000\000\251\255\
\027\000\000\000\000\000\000\000\000\000\000\000\016\000\000\000\
\000\000\013\000\022\000\022\000\095\000\095\000\016\000\000\000\
\000\000\051\000\048\000\056\000\000\000\039\255\000\000\022\000\
\052\000\053\000\022\000\077\000\044\000\000\000\000\000\000\000\
\014\255\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\238\254\045\000\
\000\000\000\000\000\000\000\000\066\000\000\000\087\000\000\000\
\000\000\000\000\096\000\081\255\109\000\240\255\107\000\042\255\
\000\000\016\000\000\000\000\000\135\000\000\000\053\000\136\000\
\000\000\095\000\044\000\138\000\024\255\117\255\095\000\093\000\
\139\000\000\000\108\000\000\000\124\000\000\000\000\000\000\000\
\000\000\140\000\137\000\000\000\000\000\143\000\000\000\000\000\
\000\000\000\000\000\000\000\000\134\000\000\000\147\000\000\000\
\000\000\146\000\011\255\000\000\113\255\148\000\142\000\000\000\
\142\000\141\000\093\000\149\000\111\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\145\255\144\000\000\000\
\000\000\150\000\041\255\000\000\151\000\142\000\000\000\148\000\
\111\000\115\255\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\095\000\153\000\000\000\096\000\070\000\187\255\000\000\
\000\000\000\000\000\000\218\255\087\000\000\000\000\000\095\000\
\000\000\014\000\000\000\000\000\140\000\087\000\000\000\140\000"

let yyrindex = "\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\155\000\000\000\
\000\000\000\000\116\255\147\255\132\000\132\000\155\000\000\000\
\000\000\186\255\000\000\156\000\000\000\000\000\000\000\116\255\
\000\000\178\255\147\255\000\000\002\255\000\000\000\000\000\000\
\004\255\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\005\255\000\000\
\000\000\000\000\006\255\000\000\000\000\000\000\211\255\000\000\
\000\000\000\000\000\000\132\000\000\000\132\000\000\000\000\000\
\000\000\000\000\000\000\000\000\159\000\000\000\209\255\000\000\
\000\000\132\000\031\255\000\000\000\000\000\000\132\000\001\255\
\000\000\000\000\005\255\000\000\083\255\000\000\000\000\000\000\
\000\000\241\255\156\255\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\051\255\000\000\000\000\000\000\000\000\079\255\000\000\
\082\255\000\000\001\255\000\000\160\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\132\000\000\000\000\000\133\000\000\000\000\000\
\160\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\132\000\246\255\000\000\000\000\132\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\132\000\
\000\000\132\000\000\000\000\000\157\000\000\000\000\000\158\000"

let yygindex = "\000\000\
\000\000\000\000\101\001\000\000\109\001\000\000\114\255\160\001\
\161\001\162\001\158\255\000\000\122\255\036\001\059\001\203\255\
\002\001\000\000\108\255\000\000\025\001\000\000\139\001\000\000\
\000\000\140\001\000\000\088\001\000\000\106\000\018\001\000\000\
\000\000\249\255\145\001\000\000\000\000\000\000\134\001\132\001\
\000\000\154\000\000\000\024\001\000\000\255\000\026\001\047\001\
\000\000\000\000"

let yytablesize = 443
let yytable = "\102\000\
\026\000\010\000\159\000\111\000\110\000\011\000\174\000\007\000\
\004\000\013\000\091\000\001\000\151\000\007\000\004\000\013\000\
\111\000\003\000\007\000\004\000\013\000\152\000\007\000\159\000\
\007\000\004\000\008\000\181\000\012\000\013\000\014\000\106\000\
\056\000\015\000\129\000\016\000\017\000\174\000\130\000\092\000\
\094\000\190\000\091\000\094\000\102\000\102\000\111\000\169\000\
\107\000\102\000\101\000\007\000\058\000\092\000\170\000\054\000\
\122\000\123\000\055\000\056\000\057\000\058\000\058\000\059\000\
\060\000\061\000\062\000\063\000\064\000\065\000\066\000\067\000\
\068\000\069\000\171\000\071\000\012\000\013\000\072\000\073\000\
\042\000\042\000\189\000\041\000\041\000\042\000\011\000\031\000\
\041\000\117\000\007\000\192\000\011\000\042\000\032\000\054\000\
\041\000\011\000\055\000\056\000\057\000\172\000\058\000\059\000\
\060\000\061\000\062\000\063\000\064\000\065\000\066\000\067\000\
\068\000\069\000\070\000\071\000\012\000\013\000\072\000\073\000\
\185\000\179\000\009\000\153\000\095\000\006\000\133\000\095\000\
\154\000\092\000\095\000\134\000\172\000\095\000\095\000\095\000\
\095\000\095\000\095\000\095\000\095\000\095\000\095\000\095\000\
\095\000\095\000\095\000\095\000\095\000\095\000\095\000\095\000\
\095\000\095\000\095\000\097\000\097\000\046\000\046\000\165\000\
\166\000\097\000\046\000\029\000\097\000\097\000\097\000\097\000\
\097\000\097\000\097\000\097\000\097\000\097\000\097\000\097\000\
\097\000\097\000\097\000\097\000\097\000\097\000\097\000\097\000\
\097\000\097\000\089\000\070\000\186\000\118\000\028\000\118\000\
\093\000\187\000\070\000\093\000\093\000\093\000\093\000\093\000\
\093\000\093\000\093\000\093\000\093\000\093\000\093\000\093\000\
\093\000\093\000\093\000\093\000\093\000\093\000\093\000\093\000\
\093\000\090\000\030\000\125\000\044\000\033\000\035\000\093\000\
\188\000\044\000\093\000\093\000\093\000\093\000\093\000\093\000\
\093\000\093\000\093\000\093\000\093\000\093\000\093\000\093\000\
\093\000\093\000\093\000\093\000\093\000\093\000\093\000\093\000\
\120\000\029\000\045\000\127\000\036\000\040\000\054\000\045\000\
\139\000\055\000\056\000\057\000\029\000\058\000\059\000\060\000\
\061\000\062\000\063\000\064\000\065\000\066\000\067\000\068\000\
\069\000\070\000\071\000\012\000\013\000\072\000\073\000\101\000\
\037\000\038\000\039\000\170\000\054\000\041\000\042\000\055\000\
\056\000\057\000\011\000\058\000\059\000\060\000\061\000\062\000\
\063\000\064\000\065\000\066\000\067\000\068\000\069\000\070\000\
\071\000\012\000\013\000\072\000\073\000\101\000\046\000\088\000\
\089\000\090\000\054\000\032\000\096\000\055\000\056\000\057\000\
\097\000\058\000\059\000\060\000\061\000\062\000\063\000\064\000\
\065\000\066\000\067\000\068\000\069\000\070\000\071\000\012\000\
\013\000\072\000\073\000\184\000\054\000\100\000\112\000\055\000\
\056\000\057\000\113\000\058\000\059\000\060\000\061\000\062\000\
\063\000\064\000\065\000\066\000\067\000\068\000\069\000\070\000\
\071\000\012\000\013\000\072\000\073\000\054\000\115\000\119\000\
\055\000\056\000\057\000\121\000\058\000\059\000\060\000\061\000\
\062\000\063\000\064\000\065\000\066\000\067\000\068\000\069\000\
\070\000\071\000\012\000\013\000\072\000\073\000\043\000\043\000\
\125\000\140\000\126\000\043\000\128\000\142\000\143\000\145\000\
\144\000\147\000\133\000\043\000\148\000\149\000\150\000\157\000\
\007\000\155\000\162\000\160\000\163\000\108\000\130\000\182\000\
\168\000\082\000\107\000\066\000\067\000\109\000\154\000\018\000\
\019\000\020\000\158\000\051\000\050\000\146\000\183\000\167\000\
\086\000\124\000\087\000\180\000\051\000\095\000\099\000\177\000\
\191\000\161\000\178\000"

let yycheck = "\053\000\
\008\000\009\001\137\000\003\001\023\001\013\001\155\000\004\001\
\004\001\004\001\009\001\001\000\002\001\010\001\010\001\010\001\
\035\001\040\001\015\001\015\001\015\001\011\001\021\001\158\000\
\021\001\021\001\021\001\170\000\036\001\037\001\038\001\018\001\
\019\001\041\001\011\001\043\001\044\001\186\000\015\001\009\001\
\048\000\184\000\004\001\051\000\098\000\099\000\046\001\007\001\
\035\001\103\000\010\001\021\001\002\001\015\001\014\001\015\001\
\015\001\016\001\018\001\019\001\020\001\011\001\022\001\023\001\
\024\001\025\001\026\001\027\001\028\001\029\001\030\001\031\001\
\032\001\033\001\034\001\035\001\036\001\037\001\038\001\039\001\
\002\001\003\001\181\000\002\001\003\001\007\001\004\001\008\001\
\007\001\009\001\003\001\190\000\010\001\015\001\015\001\015\001\
\015\001\015\001\018\001\019\001\020\001\155\000\022\001\023\001\
\024\001\025\001\026\001\027\001\028\001\029\001\030\001\031\001\
\032\001\033\001\034\001\035\001\036\001\037\001\038\001\039\001\
\174\000\007\001\000\000\011\001\009\001\008\001\010\001\012\001\
\016\001\015\001\015\001\015\001\186\000\018\001\019\001\020\001\
\021\001\022\001\023\001\024\001\025\001\026\001\027\001\028\001\
\029\001\030\001\031\001\032\001\033\001\034\001\035\001\036\001\
\037\001\038\001\039\001\009\001\010\001\002\001\003\001\015\001\
\016\001\015\001\007\001\015\001\018\001\019\001\020\001\021\001\
\022\001\023\001\024\001\025\001\026\001\027\001\028\001\029\001\
\030\001\031\001\032\001\033\001\034\001\035\001\036\001\037\001\
\038\001\039\001\009\001\002\001\002\001\084\000\017\001\086\000\
\015\001\007\001\009\001\018\001\019\001\020\001\021\001\022\001\
\023\001\024\001\025\001\026\001\027\001\028\001\029\001\030\001\
\031\001\032\001\033\001\034\001\035\001\036\001\037\001\038\001\
\039\001\009\001\015\001\002\001\010\001\017\001\008\001\015\001\
\007\001\015\001\018\001\019\001\020\001\021\001\022\001\023\001\
\024\001\025\001\026\001\027\001\028\001\029\001\030\001\031\001\
\032\001\033\001\034\001\035\001\036\001\037\001\038\001\039\001\
\009\001\004\001\010\001\098\000\008\001\003\001\015\001\015\001\
\103\000\018\001\019\001\020\001\015\001\022\001\023\001\024\001\
\025\001\026\001\027\001\028\001\029\001\030\001\031\001\032\001\
\033\001\034\001\035\001\036\001\037\001\038\001\039\001\010\001\
\008\001\008\001\008\001\014\001\015\001\003\001\015\001\018\001\
\019\001\020\001\013\001\022\001\023\001\024\001\025\001\026\001\
\027\001\028\001\029\001\030\001\031\001\032\001\033\001\034\001\
\035\001\036\001\037\001\038\001\039\001\010\001\042\001\005\001\
\009\001\002\001\015\001\015\001\009\001\018\001\019\001\020\001\
\012\001\022\001\023\001\024\001\025\001\026\001\027\001\028\001\
\029\001\030\001\031\001\032\001\033\001\034\001\035\001\036\001\
\037\001\038\001\039\001\014\001\015\001\009\001\021\001\018\001\
\019\001\020\001\004\001\022\001\023\001\024\001\025\001\026\001\
\027\001\028\001\029\001\030\001\031\001\032\001\033\001\034\001\
\035\001\036\001\037\001\038\001\039\001\015\001\015\001\003\001\
\018\001\019\001\020\001\009\001\022\001\023\001\024\001\025\001\
\026\001\027\001\028\001\029\001\030\001\031\001\032\001\033\001\
\034\001\035\001\036\001\037\001\038\001\039\001\002\001\003\001\
\002\001\045\001\003\001\007\001\003\001\003\001\035\001\004\001\
\021\001\003\001\010\001\015\001\015\001\003\001\005\001\010\001\
\021\001\006\001\006\001\015\001\046\001\057\000\015\001\007\001\
\011\001\003\001\003\001\009\001\009\001\057\000\016\001\008\000\
\008\000\008\000\135\000\015\001\015\001\115\000\173\000\151\000\
\038\000\090\000\039\000\162\000\036\000\048\000\051\000\160\000\
\186\000\139\000\161\000"

let yynames_const = "\
  EOF\000\
  TDot\000\
  TComma\000\
  TSemicolon\000\
  TPtr\000\
  TEqual\000\
  TLParen\000\
  TRParen\000\
  TLBrace\000\
  TRBrace\000\
  TLBrack\000\
  TRBrack\000\
  Tpublic\000\
  Tinclude\000\
  Tconst\000\
  Tchar\000\
  Tshort\000\
  Tunsigned\000\
  Tint\000\
  Tfloat\000\
  Tdouble\000\
  Tint8\000\
  Tint16\000\
  Tint32\000\
  Tint64\000\
  Tuint8\000\
  Tuint16\000\
  Tuint32\000\
  Tuint64\000\
  Tsizet\000\
  Twchar\000\
  Tvoid\000\
  Tlong\000\
  Tstruct\000\
  Tunion\000\
  Tenum\000\
  TCudaStream\000\
  Tenclave\000\
  Tfrom\000\
  Timport\000\
  Ttrusted\000\
  Tuntrusted\000\
  Tallow\000\
  Tpropagate_errno\000\
  "

let yynames_block = "\
  Tidentifier\000\
  Tnumber\000\
  Tstring\000\
  "

let yyact = [|
  (fun _ -> failwith "parser")
; (fun __caml_parser_env ->
    Obj.repr(
# 283 "Parser.mly"
                 ( Ast.Char Ast.Signed )
# 637 "Parser.ml"
               : 'char_type))
; (fun __caml_parser_env ->
    Obj.repr(
# 284 "Parser.mly"
                    ( Ast.Char Ast.Unsigned )
# 643 "Parser.ml"
               : 'char_type))
; (fun __caml_parser_env ->
    Obj.repr(
# 288 "Parser.mly"
                     ( Ast.IShort )
# 649 "Parser.ml"
               : 'ex_shortness))
; (fun __caml_parser_env ->
    Obj.repr(
# 289 "Parser.mly"
          ( Ast.ILong )
# 655 "Parser.ml"
               : 'ex_shortness))
; (fun __caml_parser_env ->
    Obj.repr(
# 292 "Parser.mly"
                          ( Ast.LLong Ast.Signed )
# 661 "Parser.ml"
               : 'longlong))
; (fun __caml_parser_env ->
    Obj.repr(
# 293 "Parser.mly"
                          ( Ast.LLong Ast.Unsigned )
# 667 "Parser.ml"
               : 'longlong))
; (fun __caml_parser_env ->
    Obj.repr(
# 295 "Parser.mly"
                       ( Ast.INone )
# 673 "Parser.ml"
               : 'shortness))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'ex_shortness) in
    Obj.repr(
# 296 "Parser.mly"
                 ( _1 )
# 680 "Parser.ml"
               : 'shortness))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'shortness) in
    Obj.repr(
# 299 "Parser.mly"
                         (
      Ast.Int { Ast.ia_signedness = Ast.Signed; Ast.ia_shortness = _1 }
    )
# 689 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'shortness) in
    Obj.repr(
# 302 "Parser.mly"
                             (
      Ast.Int { Ast.ia_signedness = Ast.Unsigned; Ast.ia_shortness = _2 }
    )
# 698 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'shortness) in
    Obj.repr(
# 305 "Parser.mly"
                        (
      Ast.Int { Ast.ia_signedness = Ast.Unsigned; Ast.ia_shortness = _2 }
    )
# 707 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'longlong) in
    Obj.repr(
# 308 "Parser.mly"
             ( _1 )
# 714 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'ex_shortness) in
    Obj.repr(
# 309 "Parser.mly"
                 (
      Ast.Int { Ast.ia_signedness = Ast.Signed; Ast.ia_shortness = _1 }
    )
# 723 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'char_type) in
    Obj.repr(
# 315 "Parser.mly"
              ( _1 )
# 730 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'int_type) in
    Obj.repr(
# 316 "Parser.mly"
              ( _1 )
# 737 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 318 "Parser.mly"
             ( Ast.Float )
# 743 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 319 "Parser.mly"
             ( Ast.Double )
# 749 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 320 "Parser.mly"
                  ( Ast.LDouble )
# 755 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 322 "Parser.mly"
             ( Ast.Int8 )
# 761 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 323 "Parser.mly"
             ( Ast.Int16 )
# 767 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 324 "Parser.mly"
             ( Ast.Int32 )
# 773 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 325 "Parser.mly"
             ( Ast.Int64 )
# 779 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 326 "Parser.mly"
             ( Ast.UInt8 )
# 785 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 327 "Parser.mly"
             ( Ast.UInt16 )
# 791 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 328 "Parser.mly"
             ( Ast.UInt32 )
# 797 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 329 "Parser.mly"
             ( Ast.UInt64 )
# 803 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 330 "Parser.mly"
             ( Ast.SizeT )
# 809 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 331 "Parser.mly"
             ( Ast.WChar )
# 815 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 332 "Parser.mly"
             ( Ast.Void )
# 821 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 334 "Parser.mly"
                  ( Ast.CudaStrm )
# 827 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'struct_specifier) in
    Obj.repr(
# 336 "Parser.mly"
                     ( _1 )
# 834 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'union_specifier) in
    Obj.repr(
# 337 "Parser.mly"
                     ( _1 )
# 841 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'enum_specifier) in
    Obj.repr(
# 338 "Parser.mly"
                     ( _1 )
# 848 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 339 "Parser.mly"
                     ( Ast.Foreign(_1) )
# 855 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 342 "Parser.mly"
                 ( fun ii -> Ast.Ptr(ii) )
# 861 "Parser.ml"
               : 'pointer))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'pointer) in
    Obj.repr(
# 343 "Parser.mly"
                 ( fun ii -> Ast.Ptr(_1 ii) )
# 868 "Parser.ml"
               : 'pointer))
; (fun __caml_parser_env ->
    Obj.repr(
# 346 "Parser.mly"
                                         ( failwith "Flexible array is not supported." )
# 874 "Parser.ml"
               : 'empty_dimension))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : int) in
    Obj.repr(
# 347 "Parser.mly"
                                         ( if _2 <> 0 then [_2]
                                           else failwith "Zero-length array is not supported." )
# 882 "Parser.ml"
               : 'fixed_dimension))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'fixed_dimension) in
    Obj.repr(
# 350 "Parser.mly"
                                     ( _1 )
# 889 "Parser.ml"
               : 'fixed_size_array))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'fixed_size_array) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'fixed_dimension) in
    Obj.repr(
# 351 "Parser.mly"
                                     ( _1 @ _2 )
# 897 "Parser.ml"
               : 'fixed_size_array))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'fixed_size_array) in
    Obj.repr(
# 354 "Parser.mly"
                                     ( _1 )
# 904 "Parser.ml"
               : 'array_size))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'empty_dimension) in
    Obj.repr(
# 355 "Parser.mly"
                                     ( _1 )
# 911 "Parser.ml"
               : 'array_size))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'empty_dimension) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'fixed_size_array) in
    Obj.repr(
# 356 "Parser.mly"
                                     ( _1 @ _2 )
# 919 "Parser.ml"
               : 'array_size))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'type_spec) in
    Obj.repr(
# 359 "Parser.mly"
                      ( _1 )
# 926 "Parser.ml"
               : 'all_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'type_spec) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'pointer) in
    Obj.repr(
# 360 "Parser.mly"
                      ( _2 _1 )
# 934 "Parser.ml"
               : 'all_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 363 "Parser.mly"
                           ( { Ast.identifier = _1; Ast.array_dims = []; } )
# 941 "Parser.ml"
               : 'declarator))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'array_size) in
    Obj.repr(
# 364 "Parser.mly"
                           ( { Ast.identifier = _1; Ast.array_dims = _2; } )
# 949 "Parser.ml"
               : 'declarator))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'attr_block) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'all_type) in
    Obj.repr(
# 373 "Parser.mly"
                                (
    match _2 with
      Ast.Ptr _ -> fun x -> Ast.PTPtr(_2, get_ptr_attr _1)
    | _         ->
      if _1 <> [] then
        let attr = get_ptr_attr _1 in
        match _2 with
          Ast.Foreign s ->
            if attr.Ast.pa_isptr || attr.Ast.pa_isary then fun x -> Ast.PTPtr(_2, attr)
            else
              (* thinking about 'user_defined_type var[4]' *)
              fun is_ary ->
                if is_ary then Ast.PTPtr(_2, attr)
                else failwithf "`%s' is considerred plain type but decorated with pointer attributes" s
        | _ ->
          fun is_ary ->
            if is_ary then Ast.PTPtr(_2, attr)
            else failwithf "unexpected pointer attributes for `%s'" (Ast.get_tystr _2)
      else
        fun is_ary ->
          if is_ary then Ast.PTPtr(_2, get_ptr_attr [])
          else  Ast.PTVal _2
    )
# 979 "Parser.ml"
               : 'param_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'all_type) in
    Obj.repr(
# 396 "Parser.mly"
             (
    match _1 with
      Ast.Ptr _ -> fun x -> Ast.PTPtr(_1, get_ptr_attr [])
    | _         ->
      fun is_ary ->
        if is_ary then Ast.PTPtr(_1, get_ptr_attr [])
        else  Ast.PTVal _1
    )
# 993 "Parser.ml"
               : 'param_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'attr_block) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'type_spec) in
    let _4 = (Parsing.peek_val __caml_parser_env 0 : 'pointer) in
    Obj.repr(
# 404 "Parser.mly"
                                        (
      let attr = get_ptr_attr _1
      in fun x -> Ast.PTPtr(_4 _3, { attr with Ast.pa_rdonly = true })
    )
# 1005 "Parser.ml"
               : 'param_type))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'type_spec) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'pointer) in
    Obj.repr(
# 408 "Parser.mly"
                             (
      let attr = get_ptr_attr []
      in fun x -> Ast.PTPtr(_3 _2, { attr with Ast.pa_rdonly = true })
    )
# 1016 "Parser.ml"
               : 'param_type))
; (fun __caml_parser_env ->
    Obj.repr(
# 415 "Parser.mly"
                                  ( failwith "no attribute specified." )
# 1022 "Parser.ml"
               : 'attr_block))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'key_val_pairs) in
    Obj.repr(
# 416 "Parser.mly"
                                  ( _2 )
# 1029 "Parser.ml"
               : 'attr_block))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'key_val_pair) in
    Obj.repr(
# 419 "Parser.mly"
                                      ( [_1] )
# 1036 "Parser.ml"
               : 'key_val_pairs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'key_val_pairs) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'key_val_pair) in
    Obj.repr(
# 420 "Parser.mly"
                                      (  _3 :: _1 )
# 1044 "Parser.ml"
               : 'key_val_pairs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 423 "Parser.mly"
                                             ( (_1, Ast.AString(_3)) )
# 1052 "Parser.ml"
               : 'key_val_pair))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : int) in
    Obj.repr(
# 424 "Parser.mly"
                                             ( (_1, Ast.ANumber(_3)) )
# 1060 "Parser.ml"
               : 'key_val_pair))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 425 "Parser.mly"
                                             ( (_1, Ast.AString("")) )
# 1067 "Parser.ml"
               : 'key_val_pair))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 428 "Parser.mly"
                                      ( Ast.Struct(_2) )
# 1074 "Parser.ml"
               : 'struct_specifier))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 429 "Parser.mly"
                                      ( Ast.Union(_2) )
# 1081 "Parser.ml"
               : 'union_specifier))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 430 "Parser.mly"
                                      ( Ast.Enum(_2) )
# 1088 "Parser.ml"
               : 'enum_specifier))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'struct_specifier) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'member_list) in
    Obj.repr(
# 432 "Parser.mly"
                                                                (
    let s = { Ast.sname = (match _1 with Ast.Struct s -> s | _ -> "");
              Ast.mlist = List.rev _3; }
    in Ast.StructDef(s)
  )
# 1100 "Parser.ml"
               : 'struct_definition))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'union_specifier) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'member_list) in
    Obj.repr(
# 438 "Parser.mly"
                                                              (
    let s = { Ast.sname = (match _1 with Ast.Union s -> s | _ -> "");
              Ast.mlist = List.rev _3; }
    in Ast.UnionDef(s)
  )
# 1112 "Parser.ml"
               : 'union_definition))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'enum_body) in
    Obj.repr(
# 445 "Parser.mly"
                                                 (
      let e = { Ast.enname = ""; Ast.enbody = _3; }
      in Ast.EnumDef(e)
    )
# 1122 "Parser.ml"
               : 'enum_definition))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'enum_specifier) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'enum_body) in
    Obj.repr(
# 449 "Parser.mly"
                                             (
      let e = { Ast.enname = (match _1 with Ast.Enum s -> s | _ -> "");
                Ast.enbody = _3; }
      in Ast.EnumDef(e)
    )
# 1134 "Parser.ml"
               : 'enum_definition))
; (fun __caml_parser_env ->
    Obj.repr(
# 456 "Parser.mly"
                       ( [] )
# 1140 "Parser.ml"
               : 'enum_body))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'enum_eles) in
    Obj.repr(
# 457 "Parser.mly"
                       ( List.rev _1 )
# 1147 "Parser.ml"
               : 'enum_body))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'enum_ele) in
    Obj.repr(
# 460 "Parser.mly"
                              ( [_1] )
# 1154 "Parser.ml"
               : 'enum_eles))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'enum_eles) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'enum_ele) in
    Obj.repr(
# 461 "Parser.mly"
                              ( _3 :: _1 )
# 1162 "Parser.ml"
               : 'enum_eles))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 464 "Parser.mly"
                                   ( (_1, Ast.EnumValNone) )
# 1169 "Parser.ml"
               : 'enum_ele))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 465 "Parser.mly"
                                   ( (_1, Ast.EnumVal (Ast.AString _3)) )
# 1177 "Parser.ml"
               : 'enum_ele))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : int) in
    Obj.repr(
# 466 "Parser.mly"
                                   ( (_1, Ast.EnumVal (Ast.ANumber _3)) )
# 1185 "Parser.ml"
               : 'enum_ele))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'struct_definition) in
    Obj.repr(
# 469 "Parser.mly"
                                      ( _1 )
# 1192 "Parser.ml"
               : 'composite_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'union_definition) in
    Obj.repr(
# 470 "Parser.mly"
                                      ( _1 )
# 1199 "Parser.ml"
               : 'composite_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'enum_definition) in
    Obj.repr(
# 471 "Parser.mly"
                                      ( _1 )
# 1206 "Parser.ml"
               : 'composite_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'member_def) in
    Obj.repr(
# 474 "Parser.mly"
                                      ( [_1] )
# 1213 "Parser.ml"
               : 'member_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'member_list) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'member_def) in
    Obj.repr(
# 475 "Parser.mly"
                                      ( _2 :: _1 )
# 1221 "Parser.ml"
               : 'member_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'all_type) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'declarator) in
    Obj.repr(
# 478 "Parser.mly"
                                ( (_1, _2) )
# 1229 "Parser.ml"
               : 'member_def))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 483 "Parser.mly"
                                  ( [_1] )
# 1236 "Parser.ml"
               : 'func_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'func_list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 484 "Parser.mly"
                                  ( _3 :: _1 )
# 1244 "Parser.ml"
               : 'func_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 487 "Parser.mly"
                                  ( _1 )
# 1251 "Parser.ml"
               : 'module_path))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 2 : 'module_path) in
    let _4 = (Parsing.peek_val __caml_parser_env 0 : 'func_list) in
    Obj.repr(
# 489 "Parser.mly"
                                                         (
      { Ast.mname = _2; Ast.flist = List.rev _4; }
    )
# 1261 "Parser.ml"
               : 'import_declaration))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 2 : 'module_path) in
    Obj.repr(
# 492 "Parser.mly"
                                   (
      { Ast.mname = _2; Ast.flist = ["*"]; }
    )
# 1270 "Parser.ml"
               : 'import_declaration))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 497 "Parser.mly"
                                      ( _2 )
# 1277 "Parser.ml"
               : 'include_declaration))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'include_declaration) in
    Obj.repr(
# 499 "Parser.mly"
                                             ( [_1] )
# 1284 "Parser.ml"
               : 'include_declarations))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'include_declarations) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'include_declaration) in
    Obj.repr(
# 500 "Parser.mly"
                                             ( _2 :: _1 )
# 1292 "Parser.ml"
               : 'include_declarations))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 2 : 'trusted_block) in
    Obj.repr(
# 506 "Parser.mly"
                                                                     (
      List.rev _3
    )
# 1301 "Parser.ml"
               : 'enclave_functions))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 2 : 'untrusted_block) in
    Obj.repr(
# 509 "Parser.mly"
                                                          (
      List.rev _3
    )
# 1310 "Parser.ml"
               : 'enclave_functions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'trusted_functions) in
    Obj.repr(
# 514 "Parser.mly"
                                             ( _1 )
# 1317 "Parser.ml"
               : 'trusted_block))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'include_declarations) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'trusted_functions) in
    Obj.repr(
# 515 "Parser.mly"
                                             (
      trusted_headers := !trusted_headers @ List.rev _1; _2
    )
# 1327 "Parser.ml"
               : 'trusted_block))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'untrusted_functions) in
    Obj.repr(
# 520 "Parser.mly"
                                             ( _1 )
# 1334 "Parser.ml"
               : 'untrusted_block))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'include_declarations) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'untrusted_functions) in
    Obj.repr(
# 521 "Parser.mly"
                                             (
      untrusted_headers := !untrusted_headers @ List.rev _1; _2
    )
# 1344 "Parser.ml"
               : 'untrusted_block))
; (fun __caml_parser_env ->
    Obj.repr(
# 527 "Parser.mly"
                               ( true )
# 1350 "Parser.ml"
               : 'access_modifier))
; (fun __caml_parser_env ->
    Obj.repr(
# 528 "Parser.mly"
                               ( false  )
# 1356 "Parser.ml"
               : 'access_modifier))
; (fun __caml_parser_env ->
    Obj.repr(
# 531 "Parser.mly"
                                          ( [] )
# 1362 "Parser.ml"
               : 'trusted_functions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'trusted_functions) in
    let _2 = (Parsing.peek_val __caml_parser_env 2 : 'access_modifier) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'func_def) in
    Obj.repr(
# 532 "Parser.mly"
                                                          (
      check_ptr_attr _3 (symbol_start_pos(), symbol_end_pos());
      Ast.Trusted { Ast.tf_fdecl = _3; Ast.tf_is_priv = _2 } :: _1
    )
# 1374 "Parser.ml"
               : 'trusted_functions))
; (fun __caml_parser_env ->
    Obj.repr(
# 538 "Parser.mly"
                                                      ( [] )
# 1380 "Parser.ml"
               : 'untrusted_functions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'untrusted_functions) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'untrusted_func_def) in
    Obj.repr(
# 539 "Parser.mly"
                                                      ( _2 :: _1 )
# 1388 "Parser.ml"
               : 'untrusted_functions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'all_type) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'parameter_list) in
    Obj.repr(
# 542 "Parser.mly"
                                              (
      { Ast.fname = _2; Ast.rtype = _1; Ast.plist = List.rev _3 ; }
    )
# 1399 "Parser.ml"
               : 'func_def))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'all_type) in
    let _2 = (Parsing.peek_val __caml_parser_env 2 : 'array_size) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : string) in
    let _4 = (Parsing.peek_val __caml_parser_env 0 : 'parameter_list) in
    Obj.repr(
# 545 "Parser.mly"
                                                   (
      failwithf "%s: returning an array is not supported - use pointer instead." _3
    )
# 1411 "Parser.ml"
               : 'func_def))
; (fun __caml_parser_env ->
    Obj.repr(
# 550 "Parser.mly"
                                   ( [] )
# 1417 "Parser.ml"
               : 'parameter_list))
; (fun __caml_parser_env ->
    Obj.repr(
# 551 "Parser.mly"
                                   ( [] )
# 1423 "Parser.ml"
               : 'parameter_list))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'parameter_defs) in
    Obj.repr(
# 552 "Parser.mly"
                                   ( _2 )
# 1430 "Parser.ml"
               : 'parameter_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'parameter_def) in
    Obj.repr(
# 555 "Parser.mly"
                                        ( [_1] )
# 1437 "Parser.ml"
               : 'parameter_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'parameter_defs) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'parameter_def) in
    Obj.repr(
# 556 "Parser.mly"
                                        ( _3 :: _1 )
# 1445 "Parser.ml"
               : 'parameter_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'param_type) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'declarator) in
    Obj.repr(
# 559 "Parser.mly"
                                     (
    let pt = _1 (Ast.is_array _2) in
    let is_void =
      match pt with
          Ast.PTVal v -> v = Ast.Void
        | _           -> false
    in
      if is_void then
        failwithf "parameter `%s' has `void' type." _2.Ast.identifier
      else
        (pt, _2)
  )
# 1464 "Parser.ml"
               : 'parameter_def))
; (fun __caml_parser_env ->
    Obj.repr(
# 573 "Parser.mly"
                               ( false )
# 1470 "Parser.ml"
               : 'propagate_errno))
; (fun __caml_parser_env ->
    Obj.repr(
# 574 "Parser.mly"
                               ( true  )
# 1476 "Parser.ml"
               : 'propagate_errno))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'attr_block) in
    let _2 = (Parsing.peek_val __caml_parser_env 2 : 'func_def) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'allow_list) in
    let _4 = (Parsing.peek_val __caml_parser_env 0 : 'propagate_errno) in
    Obj.repr(
# 577 "Parser.mly"
                                                                   (
      check_ptr_attr _2 (symbol_start_pos(), symbol_end_pos());
      let fattr = get_func_attr _1 in
      Ast.Untrusted { Ast.uf_fdecl = _2; Ast.uf_fattr = fattr; Ast.uf_allow_list = _3; Ast.uf_propagate_errno = _4 }
    )
# 1490 "Parser.ml"
               : 'untrusted_func_def))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'func_def) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'allow_list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'propagate_errno) in
    Obj.repr(
# 582 "Parser.mly"
                                        (
      check_ptr_attr _1 (symbol_start_pos(), symbol_end_pos());
      let fattr = get_func_attr [] in
      Ast.Untrusted { Ast.uf_fdecl = _1; Ast.uf_fattr = fattr; Ast.uf_allow_list = _2; Ast.uf_propagate_errno = _3 }
    )
# 1503 "Parser.ml"
               : 'untrusted_func_def))
; (fun __caml_parser_env ->
    Obj.repr(
# 589 "Parser.mly"
                                     ( [] )
# 1509 "Parser.ml"
               : 'allow_list))
; (fun __caml_parser_env ->
    Obj.repr(
# 590 "Parser.mly"
                                     ( [] )
# 1515 "Parser.ml"
               : 'allow_list))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'func_list) in
    Obj.repr(
# 591 "Parser.mly"
                                     ( _3 )
# 1522 "Parser.ml"
               : 'allow_list))
; (fun __caml_parser_env ->
    Obj.repr(
# 597 "Parser.mly"
                           ( [] )
# 1528 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'expressions) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'include_declaration) in
    Obj.repr(
# 598 "Parser.mly"
                                              ( Ast.Include(_2)   :: _1 )
# 1536 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'expressions) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'import_declaration) in
    Obj.repr(
# 599 "Parser.mly"
                                              ( Ast.Importing(_2) :: _1 )
# 1544 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'expressions) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'composite_defs) in
    Obj.repr(
# 600 "Parser.mly"
                                              ( Ast.Composite(_2) :: _1 )
# 1552 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'expressions) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'enclave_functions) in
    Obj.repr(
# 601 "Parser.mly"
                                              ( Ast.Interface(_2) :: _1 )
# 1560 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'expressions) in
    Obj.repr(
# 604 "Parser.mly"
                                                  (
      { Ast.ename = "";
        Ast.eexpr = List.rev _3 }
    )
# 1570 "Parser.ml"
               : 'enclave_def))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'enclave_def) in
    Obj.repr(
# 613 "Parser.mly"
                                          ( _1 )
# 1577 "Parser.ml"
               : Ast.enclave))
(* Entry start_parsing *)
; (fun __caml_parser_env -> raise (Parsing.YYexit (Parsing.peek_val __caml_parser_env 0)))
|]
let yytables =
  { Parsing.actions=yyact;
    Parsing.transl_const=yytransl_const;
    Parsing.transl_block=yytransl_block;
    Parsing.lhs=yylhs;
    Parsing.len=yylen;
    Parsing.defred=yydefred;
    Parsing.dgoto=yydgoto;
    Parsing.sindex=yysindex;
    Parsing.rindex=yyrindex;
    Parsing.gindex=yygindex;
    Parsing.tablesize=yytablesize;
    Parsing.table=yytable;
    Parsing.check=yycheck;
    Parsing.error_function=parse_error;
    Parsing.names_const=yynames_const;
    Parsing.names_block=yynames_block }
let start_parsing (lexfun : Lexing.lexbuf -> token) (lexbuf : Lexing.lexbuf) =
   (Parsing.yyparse yytables 1 lexfun lexbuf : Ast.enclave)
;;
