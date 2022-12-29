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
  | Tswitchless
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

(* Check whether 'size' is specified. *)
let has_size (sattr: Ast.ptr_size) =
  sattr.Ast.ps_size <> None
  
(* Check whether 'count' is specified. *)
let has_count (sattr: Ast.ptr_size) =
  sattr.Ast.ps_count <> None

(* Pointers can have the following attributes:
 *
 * 'size'     - specifies the size of the pointer.
 *              e.g. size = 4, size = val ('val' is a parameter);
 *
 * 'count'    - indicates how many of items is managed by the pointer
 *              e.g. count = 100, count = n ('n' is a parameter);
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
 * Note that 'size' can be used together with 'count'.
 * 'string' and 'wstring' indicates 'isptr',
 * and they cannot be used with only an 'out' attribute.
 *)
let get_ptr_attr (attr_list: (string * Ast.attr_value) list) =
  let get_new_dir (cds: string) (cda: Ast.ptr_direction) (old: Ast.ptr_direction) =
    if old = Ast.PtrNoDirection then cda
    else if old = Ast.PtrInOut  then failwithf "duplicated attribute: `%s'" cds
    else if old = cda           then failwithf "duplicated attribute: `%s'" cds
    else Ast.PtrInOut
  in
  (* only one 'size' attribute allowed. *)
  let get_new_size (new_value: Ast.attr_value) (old_ptr_size: Ast.ptr_size) =
    if has_size old_ptr_size then
     failwithf "duplicated attribute: `size'"
    else new_value
  in
  (* only one 'count' attribute allowed. *)
  let get_new_count (new_value: Ast.attr_value) (old_ptr_size: Ast.ptr_size) =
    if has_count old_ptr_size then
      failwithf "duplicated attribute: `count'"
    else new_value
  in
  let update_attr (key: string) (value: Ast.attr_value) (res: Ast.ptr_attr) =
    match key with
        "size"     ->
        { res with Ast.pa_size = { res.Ast.pa_size with Ast.ps_size  = Some(get_new_size value res.Ast.pa_size)}}
      | "count"    ->
        { res with Ast.pa_size = { res.Ast.pa_size with Ast.ps_count = Some(get_new_count value res.Ast.pa_size)}}
      | "sizefunc" ->
        failwithf "The attribute 'sizefunc' is deprecated. Please use 'size' attribute instead."
      | "string"  -> { res with Ast.pa_isstr = true; }
      | "wstring" -> { res with Ast.pa_iswstr = true; }
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
      | _ -> failwithf "unknown attribute: %s" key
  in
  let rec do_get_ptr_attr alist res_attr =
    match alist with
        [] -> res_attr
      | (k,v) :: xs -> do_get_ptr_attr xs (update_attr k v res_attr)
  in
    do_get_ptr_attr attr_list            { Ast.pa_direction = Ast.PtrNoDirection;
                                           Ast.pa_size = Ast.empty_ptr_size;
                                           Ast.pa_isptr = false;
                                           Ast.pa_isary = false;
                                           Ast.pa_isstr = false;
                                           Ast.pa_iswstr = false;
                                           Ast.pa_rdonly = false;
                                           Ast.pa_chkptr = true;
                                         }

let get_param_ptr_attr (attr_list: (string * Ast.attr_value) list) =
  let has_str_attr (pattr: Ast.ptr_attr) =
    if pattr.Ast.pa_isstr && pattr.Ast.pa_iswstr
    then failwith "`string' and `wstring' are mutual exclusive"
    else (pattr.Ast.pa_isstr || pattr.Ast.pa_iswstr)
  in
  let check_invalid_ptr_size (pattr: Ast.ptr_attr) =
    let ps = pattr.Ast.pa_size in
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
        if pattr.Ast.pa_direction = Ast.PtrOut && has_str_attr pattr
        then failwith "string/wstring should be used with an `in' attribute"
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
  let pattr = get_ptr_attr attr_list in
  if pattr.Ast.pa_isary
  then check_invalid_ary_attr pattr
  else check_invalid_ptr_size pattr |> check_ptr_dir

    
let get_member_ptr_attr (attr_list: (string * Ast.attr_value) list) =
  let check_invalid_ptr_size (pattr: Ast.ptr_attr) =
          if pattr.Ast.pa_size = Ast.empty_ptr_size
          then failwith "size/count attributes must be used"
          else pattr
  in
  let pattr = get_ptr_attr attr_list in
  check_invalid_ptr_size pattr

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
    if pattr.Ast.pa_chkptr && (not (has_size pattr.Ast.pa_size))
    then failwithf "`%s': void pointer `%s' - buffer size unknown" fname identifier
    else ()
  in
  let check_string_ptr_size (atype: Ast.atype) (pattr: Ast.ptr_attr) (identifier: string) =
    if (pattr.Ast.pa_isstr)
    then
      match atype with
      Ast.Ptr(Ast.Char(_)) -> ()
      | _ -> failwithf "`%s': invalid 'string' attribute - `%s' is not char pointer." fname identifier
    else
      if (atype <> Ast.Ptr(Ast.WChar) &&  pattr.Ast.pa_iswstr)
      then failwithf "`%s': invalid 'wstring' attribute - `%s' is not wchar_t pointer." fname identifier
      else ()
  in
  let check_array_dims (atype: Ast.atype) (pattr: Ast.ptr_attr) (declr: Ast.declarator) =
    if Ast.is_array declr then
      if has_size pattr.Ast.pa_size then
        failwithf "`%s': invalid 'size' attribute - `%s' is explicitly declared array." fname declr.Ast.identifier
      else if has_count pattr.Ast.pa_size then
        failwithf "`%s': invalid 'count' attribute - `%s' is explicitly declared array." fname declr.Ast.identifier
      else if pattr.Ast.pa_isary then
        failwithf "`%s': invalid 'isary' attribute - `%s' is explicitly declared array." fname declr.Ast.identifier
    else ()
  in
  let check_pointer_array (atype: Ast.atype) (pattr: Ast.ptr_attr) (declr: Ast.declarator) = 
    let is_ary = (Ast.is_array declr || pattr.Ast.pa_isary) in
    let is_ptr  =
      match atype with
        Ast.Ptr _ -> true
      | _         -> pattr.Ast.pa_isptr
    in
    if is_ary && is_ptr then
      failwithf "`%s': Pointer array not allowed - `%s' is a pointer array." fname declr.Ast.identifier 
    else ()
  in
  let checker (pd: Ast.pdecl) =
    let pt, declr = pd in
    let identifier = declr.Ast.identifier in
      match pt with
          Ast.PTVal _ -> ()
        | Ast.PTPtr(atype, pattr) ->
          if atype = Ast.Ptr(Ast.Void) then (* 'void' pointer, check there is a size or 'user_check' *)
            check_void_ptr_size pattr identifier
          else
            check_pointer_array atype pattr declr;
            check_const pattr identifier;
            check_string_ptr_size atype pattr identifier;
            check_array_dims atype pattr declr
  in
    List.iter checker fd.Ast.plist
# 324 "Parser.ml"
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
  269 (* Tswitchless *);
  270 (* Tinclude *);
  271 (* Tconst *);
  275 (* Tchar *);
  276 (* Tshort *);
  277 (* Tunsigned *);
  278 (* Tint *);
  279 (* Tfloat *);
  280 (* Tdouble *);
  281 (* Tint8 *);
  282 (* Tint16 *);
  283 (* Tint32 *);
  284 (* Tint64 *);
  285 (* Tuint8 *);
  286 (* Tuint16 *);
  287 (* Tuint32 *);
  288 (* Tuint64 *);
  289 (* Tsizet *);
  290 (* Twchar *);
  291 (* Tvoid *);
  292 (* Tlong *);
  293 (* Tstruct *);
  294 (* Tunion *);
  295 (* Tenum *);
  296 (* Tenclave *);
  297 (* Tfrom *);
  298 (* Timport *);
  299 (* Ttrusted *);
  300 (* Tuntrusted *);
  301 (* Tallow *);
  302 (* Tpropagate_errno *);
    0|]

let yytransl_block = [|
  272 (* Tidentifier *);
  273 (* Tnumber *);
  274 (* Tstring *);
    0|]

let yylhs = "\255\255\
\002\000\002\000\003\000\003\000\004\000\004\000\005\000\005\000\
\006\000\006\000\006\000\006\000\006\000\007\000\007\000\007\000\
\007\000\007\000\007\000\007\000\007\000\007\000\007\000\007\000\
\007\000\007\000\007\000\007\000\007\000\007\000\007\000\007\000\
\007\000\011\000\011\000\012\000\013\000\014\000\014\000\015\000\
\015\000\015\000\016\000\016\000\017\000\017\000\018\000\018\000\
\018\000\018\000\020\000\020\000\019\000\019\000\021\000\021\000\
\022\000\022\000\022\000\008\000\009\000\010\000\023\000\025\000\
\027\000\027\000\028\000\028\000\029\000\029\000\030\000\030\000\
\030\000\031\000\031\000\031\000\024\000\024\000\026\000\026\000\
\032\000\033\000\034\000\034\000\035\000\036\000\036\000\037\000\
\038\000\038\000\039\000\039\000\040\000\040\000\041\000\041\000\
\044\000\044\000\045\000\045\000\042\000\042\000\043\000\043\000\
\046\000\046\000\048\000\048\000\048\000\049\000\049\000\050\000\
\051\000\051\000\052\000\052\000\053\000\053\000\053\000\047\000\
\054\000\054\000\054\000\055\000\055\000\055\000\055\000\055\000\
\056\000\001\000\000\000"

let yylen = "\002\000\
\001\000\002\000\001\000\001\000\002\000\003\000\000\000\001\000\
\002\000\003\000\002\000\001\000\001\000\001\000\001\000\001\000\
\001\000\002\000\001\000\001\000\001\000\001\000\001\000\001\000\
\001\000\001\000\001\000\001\000\001\000\001\000\001\000\001\000\
\001\000\001\000\002\000\002\000\003\000\001\000\002\000\001\000\
\001\000\002\000\001\000\002\000\001\000\002\000\002\000\001\000\
\004\000\003\000\002\000\001\000\002\000\003\000\001\000\003\000\
\003\000\003\000\001\000\002\000\002\000\002\000\004\000\004\000\
\004\000\004\000\000\000\001\000\001\000\003\000\001\000\003\000\
\003\000\001\000\001\000\001\000\002\000\003\000\002\000\003\000\
\002\000\002\000\001\000\003\000\001\000\004\000\004\000\002\000\
\001\000\002\000\005\000\005\000\001\000\002\000\001\000\002\000\
\000\000\001\000\000\000\001\000\000\000\005\000\000\000\003\000\
\003\000\004\000\002\000\003\000\003\000\001\000\003\000\002\000\
\000\000\001\000\000\000\001\000\000\000\002\000\002\000\004\000\
\000\000\003\000\004\000\000\000\002\000\003\000\003\000\002\000\
\004\000\003\000\002\000"

let yydefred = "\000\000\
\000\000\000\000\000\000\131\000\000\000\124\000\000\000\000\000\
\130\000\129\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\074\000\075\000\076\000\000\000\
\000\000\125\000\128\000\088\000\060\000\061\000\000\000\062\000\
\085\000\000\000\000\000\000\000\000\000\000\000\000\000\127\000\
\126\000\000\000\000\000\000\000\069\000\000\000\089\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\033\000\001\000\
\003\000\000\000\016\000\017\000\019\000\020\000\021\000\022\000\
\023\000\024\000\025\000\026\000\027\000\028\000\029\000\000\000\
\000\000\014\000\000\000\012\000\000\000\015\000\000\000\030\000\
\031\000\032\000\052\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\065\000\000\000\087\000\083\000\
\000\000\090\000\000\000\000\000\098\000\000\000\000\000\000\000\
\116\000\000\000\000\000\053\000\000\000\000\000\055\000\002\000\
\000\000\008\000\000\000\018\000\005\000\009\000\034\000\000\000\
\051\000\000\000\081\000\063\000\000\000\077\000\082\000\064\000\
\000\000\079\000\066\000\072\000\073\000\070\000\000\000\091\000\
\000\000\000\000\092\000\104\000\000\000\000\000\000\000\054\000\
\006\000\010\000\035\000\000\000\000\000\038\000\000\000\046\000\
\078\000\080\000\084\000\000\000\000\000\100\000\000\000\000\000\
\000\000\057\000\058\000\056\000\036\000\000\000\000\000\000\000\
\039\000\000\000\105\000\000\000\102\000\000\000\000\000\000\000\
\120\000\037\000\107\000\000\000\000\000\048\000\000\000\000\000\
\000\000\110\000\106\000\122\000\000\000\114\000\119\000\118\000\
\000\000\108\000\112\000\000\000\047\000\000\000\109\000\123\000\
\000\000\000\000\111\000\000\000"

let yydgoto = "\002\000\
\004\000\074\000\075\000\076\000\077\000\078\000\079\000\080\000\
\081\000\082\000\120\000\149\000\150\000\151\000\152\000\083\000\
\123\000\183\000\084\000\085\000\110\000\111\000\021\000\086\000\
\022\000\089\000\023\000\043\000\044\000\045\000\024\000\087\000\
\090\000\097\000\034\000\025\000\047\000\048\000\027\000\049\000\
\052\000\050\000\053\000\102\000\159\000\138\000\106\000\171\000\
\185\000\186\000\191\000\107\000\177\000\161\000\008\000\005\000"

let yysindex = "\023\000\
\242\254\000\000\032\255\000\000\041\255\000\000\058\000\250\254\
\000\000\000\000\071\255\077\255\115\255\129\255\122\255\145\255\
\148\255\149\255\151\255\176\255\000\000\000\000\000\000\184\255\
\185\255\000\000\000\000\000\000\000\000\000\000\125\255\000\000\
\000\000\153\255\177\255\177\255\132\000\181\000\125\255\000\000\
\000\000\210\255\208\255\216\255\000\000\007\255\000\000\177\255\
\213\255\233\255\177\255\237\255\238\255\139\255\000\000\000\000\
\000\000\254\254\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\017\255\
\231\255\000\000\000\000\000\000\227\255\000\000\246\255\000\000\
\000\000\000\000\000\000\181\000\236\255\173\255\250\255\236\255\
\072\000\017\000\012\000\044\255\000\000\125\255\000\000\000\000\
\020\000\000\000\233\255\021\000\000\000\181\000\238\255\022\000\
\000\000\024\000\181\000\000\000\018\000\048\255\000\000\000\000\
\248\255\000\000\029\000\000\000\000\000\000\000\000\000\048\000\
\000\000\043\000\000\000\000\000\051\000\000\000\000\000\000\000\
\052\000\000\000\000\000\000\000\000\000\000\000\040\000\000\000\
\108\255\045\000\000\000\000\000\014\000\075\255\066\000\000\000\
\000\000\000\000\000\000\127\255\073\000\000\000\073\000\000\000\
\000\000\000\000\000\000\078\000\069\000\000\000\083\000\081\000\
\249\254\000\000\000\000\000\000\000\000\079\000\077\000\073\000\
\000\000\047\255\000\000\078\000\000\000\049\255\067\000\045\000\
\000\000\000\000\000\000\181\000\082\000\000\000\236\255\157\000\
\212\255\000\000\000\000\000\000\214\255\000\000\000\000\000\000\
\246\255\000\000\000\000\181\000\000\000\102\000\000\000\000\000\
\048\000\246\255\000\000\048\000"

let yyrindex = "\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\105\000\000\000\
\000\000\000\000\078\255\142\255\093\000\093\000\105\000\000\000\
\000\000\086\255\000\000\107\000\000\000\000\000\000\000\078\255\
\000\000\204\255\142\255\000\000\235\255\000\000\000\000\000\000\
\000\000\003\255\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\005\255\
\000\000\000\000\006\255\000\000\000\000\000\000\133\255\000\000\
\000\000\000\000\000\000\093\000\000\000\093\000\000\000\000\000\
\093\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\116\000\000\000\010\000\000\000\000\000\093\000\041\000\000\000\
\000\000\000\000\093\000\000\000\124\255\000\000\000\000\000\000\
\005\255\000\000\039\255\000\000\000\000\000\000\000\000\138\255\
\000\000\183\255\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\117\000\000\000\000\000\001\255\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\118\255\000\000\120\255\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\121\000\000\000\000\000\000\000\000\000\000\000\000\000\126\255\
\000\000\093\000\000\000\000\000\000\000\000\000\140\000\117\000\
\000\000\000\000\000\000\093\000\026\255\000\000\000\000\093\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\093\000\000\000\093\000\000\000\000\000\
\128\000\000\000\000\000\129\000"

let yygindex = "\000\000\
\000\000\000\000\088\001\000\000\089\001\000\000\096\255\141\001\
\142\001\146\001\193\255\000\000\117\255\025\001\038\001\218\255\
\169\255\000\000\205\255\000\000\000\000\036\001\000\000\000\000\
\000\000\000\000\000\000\159\001\000\000\105\001\000\000\117\001\
\132\001\048\001\000\000\000\000\253\255\187\001\000\000\000\000\
\000\000\176\001\174\001\000\000\050\001\120\001\000\000\056\001\
\000\000\031\001\000\000\000\000\000\000\000\000\000\000\000\000"

let yytablesize = 485
let yytable = "\088\000\
\127\000\105\000\010\000\121\000\026\000\175\000\007\000\011\000\
\004\000\013\000\095\000\169\000\007\000\121\000\004\000\013\000\
\112\000\057\000\007\000\193\000\004\000\013\000\096\000\001\000\
\007\000\003\000\004\000\008\000\169\000\029\000\012\000\013\000\
\014\000\113\000\015\000\202\000\016\000\017\000\176\000\006\000\
\116\000\029\000\011\000\007\000\098\000\121\000\121\000\098\000\
\011\000\143\000\088\000\105\000\117\000\179\000\011\000\188\000\
\054\000\009\000\144\000\132\000\133\000\180\000\055\000\137\000\
\096\000\056\000\057\000\058\000\137\000\059\000\060\000\061\000\
\062\000\063\000\064\000\065\000\066\000\067\000\068\000\069\000\
\070\000\181\000\072\000\012\000\013\000\073\000\101\000\071\000\
\028\000\101\000\162\000\163\000\029\000\101\000\071\000\195\000\
\101\000\101\000\101\000\101\000\101\000\101\000\101\000\101\000\
\101\000\101\000\101\000\101\000\101\000\101\000\101\000\101\000\
\101\000\101\000\101\000\101\000\101\000\148\000\184\000\041\000\
\041\000\040\000\040\000\156\000\041\000\059\000\040\000\042\000\
\042\000\201\000\030\000\182\000\042\000\041\000\059\000\040\000\
\031\000\165\000\204\000\033\000\042\000\042\000\043\000\166\000\
\032\000\197\000\184\000\044\000\043\000\108\000\103\000\103\000\
\035\000\044\000\109\000\036\000\037\000\103\000\038\000\182\000\
\103\000\103\000\103\000\103\000\103\000\103\000\103\000\103\000\
\103\000\103\000\103\000\103\000\103\000\103\000\103\000\103\000\
\103\000\103\000\103\000\103\000\103\000\124\000\054\000\039\000\
\045\000\045\000\040\000\041\000\055\000\045\000\011\000\056\000\
\057\000\058\000\046\000\059\000\060\000\061\000\062\000\063\000\
\064\000\065\000\066\000\067\000\068\000\069\000\070\000\071\000\
\072\000\012\000\013\000\073\000\093\000\198\000\092\000\135\000\
\093\000\094\000\199\000\097\000\200\000\100\000\097\000\097\000\
\097\000\097\000\097\000\097\000\097\000\097\000\097\000\097\000\
\097\000\097\000\097\000\097\000\097\000\097\000\097\000\097\000\
\097\000\097\000\097\000\095\000\101\000\104\000\032\000\054\000\
\118\000\119\000\115\000\122\000\126\000\115\000\115\000\115\000\
\115\000\115\000\115\000\115\000\115\000\115\000\115\000\115\000\
\115\000\115\000\115\000\115\000\115\000\115\000\115\000\115\000\
\115\000\115\000\094\000\130\000\131\000\135\000\142\000\136\000\
\139\000\097\000\140\000\145\000\097\000\097\000\097\000\097\000\
\097\000\097\000\097\000\097\000\097\000\097\000\097\000\097\000\
\097\000\097\000\097\000\097\000\097\000\097\000\097\000\097\000\
\097\000\096\000\146\000\147\000\148\000\153\000\154\000\155\000\
\115\000\158\000\160\000\115\000\115\000\115\000\115\000\115\000\
\115\000\115\000\115\000\115\000\115\000\115\000\115\000\115\000\
\115\000\115\000\115\000\115\000\115\000\115\000\115\000\115\000\
\128\000\109\000\167\000\170\000\172\000\173\000\174\000\055\000\
\194\000\178\000\056\000\057\000\058\000\166\000\059\000\060\000\
\061\000\062\000\063\000\064\000\065\000\066\000\067\000\068\000\
\069\000\070\000\071\000\072\000\012\000\013\000\073\000\054\000\
\190\000\067\000\007\000\068\000\180\000\055\000\086\000\099\000\
\056\000\057\000\058\000\117\000\059\000\060\000\061\000\062\000\
\063\000\064\000\065\000\066\000\067\000\068\000\069\000\070\000\
\071\000\072\000\012\000\013\000\073\000\054\000\113\000\050\000\
\049\000\114\000\115\000\055\000\018\000\019\000\056\000\057\000\
\058\000\020\000\059\000\060\000\061\000\062\000\063\000\064\000\
\065\000\066\000\067\000\068\000\069\000\070\000\071\000\072\000\
\012\000\013\000\073\000\196\000\055\000\168\000\157\000\056\000\
\057\000\058\000\164\000\059\000\060\000\061\000\062\000\063\000\
\064\000\065\000\066\000\067\000\068\000\069\000\070\000\071\000\
\072\000\012\000\013\000\073\000\055\000\091\000\134\000\056\000\
\057\000\058\000\125\000\059\000\060\000\061\000\062\000\063\000\
\064\000\065\000\066\000\067\000\068\000\069\000\070\000\071\000\
\072\000\012\000\013\000\073\000\129\000\189\000\051\000\099\000\
\103\000\192\000\141\000\187\000\203\000"

let yycheck = "\038\000\
\088\000\053\000\009\001\003\001\008\000\013\001\004\001\014\001\
\004\001\004\001\004\001\151\000\010\001\013\001\010\001\010\001\
\019\001\020\001\016\001\180\000\016\001\016\001\016\001\001\000\
\022\001\040\001\022\001\022\001\168\000\004\001\037\001\038\001\
\039\001\036\001\041\001\196\000\043\001\044\001\046\001\008\001\
\024\001\016\001\004\001\003\001\048\000\084\000\046\001\051\000\
\010\001\002\001\089\000\103\000\036\001\007\001\016\001\007\001\
\010\001\000\000\011\001\016\001\017\001\015\001\016\001\102\000\
\016\001\019\001\020\001\021\001\107\000\023\001\024\001\025\001\
\026\001\027\001\028\001\029\001\030\001\031\001\032\001\033\001\
\034\001\035\001\036\001\037\001\038\001\039\001\009\001\002\001\
\018\001\012\001\016\001\017\001\016\001\016\001\009\001\183\000\
\019\001\020\001\021\001\022\001\023\001\024\001\025\001\026\001\
\027\001\028\001\029\001\030\001\031\001\032\001\033\001\034\001\
\035\001\036\001\037\001\038\001\039\001\010\001\170\000\002\001\
\003\001\002\001\003\001\016\001\007\001\002\001\007\001\002\001\
\003\001\193\000\016\001\170\000\007\001\016\001\011\001\016\001\
\008\001\011\001\202\000\018\001\016\001\016\001\010\001\017\001\
\016\001\184\000\198\000\010\001\016\001\011\001\009\001\010\001\
\008\001\016\001\016\001\008\001\008\001\016\001\008\001\198\000\
\019\001\020\001\021\001\022\001\023\001\024\001\025\001\026\001\
\027\001\028\001\029\001\030\001\031\001\032\001\033\001\034\001\
\035\001\036\001\037\001\038\001\039\001\009\001\010\001\008\001\
\002\001\003\001\003\001\003\001\016\001\007\001\014\001\019\001\
\020\001\021\001\042\001\023\001\024\001\025\001\026\001\027\001\
\028\001\029\001\030\001\031\001\032\001\033\001\034\001\035\001\
\036\001\037\001\038\001\039\001\009\001\002\001\005\001\002\001\
\009\001\002\001\007\001\016\001\007\001\009\001\019\001\020\001\
\021\001\022\001\023\001\024\001\025\001\026\001\027\001\028\001\
\029\001\030\001\031\001\032\001\033\001\034\001\035\001\036\001\
\037\001\038\001\039\001\009\001\012\001\009\001\016\001\010\001\
\022\001\004\001\016\001\016\001\003\001\019\001\020\001\021\001\
\022\001\023\001\024\001\025\001\026\001\027\001\028\001\029\001\
\030\001\031\001\032\001\033\001\034\001\035\001\036\001\037\001\
\038\001\039\001\009\001\003\001\009\001\002\001\005\001\003\001\
\003\001\016\001\003\001\036\001\019\001\020\001\021\001\022\001\
\023\001\024\001\025\001\026\001\027\001\028\001\029\001\030\001\
\031\001\032\001\033\001\034\001\035\001\036\001\037\001\038\001\
\039\001\009\001\022\001\004\001\010\001\003\001\003\001\016\001\
\016\001\013\001\045\001\019\001\020\001\021\001\022\001\023\001\
\024\001\025\001\026\001\027\001\028\001\029\001\030\001\031\001\
\032\001\033\001\034\001\035\001\036\001\037\001\038\001\039\001\
\009\001\016\001\010\001\006\001\016\001\003\001\006\001\016\001\
\007\001\011\001\019\001\020\001\021\001\017\001\023\001\024\001\
\025\001\026\001\027\001\028\001\029\001\030\001\031\001\032\001\
\033\001\034\001\035\001\036\001\037\001\038\001\039\001\010\001\
\046\001\009\001\022\001\009\001\015\001\016\001\003\001\003\001\
\019\001\020\001\021\001\003\001\023\001\024\001\025\001\026\001\
\027\001\028\001\029\001\030\001\031\001\032\001\033\001\034\001\
\035\001\036\001\037\001\038\001\039\001\010\001\003\001\016\001\
\016\001\058\000\058\000\016\001\008\000\008\000\019\001\020\001\
\021\001\008\000\023\001\024\001\025\001\026\001\027\001\028\001\
\029\001\030\001\031\001\032\001\033\001\034\001\035\001\036\001\
\037\001\038\001\039\001\015\001\016\001\149\000\137\000\019\001\
\020\001\021\001\143\000\023\001\024\001\025\001\026\001\027\001\
\028\001\029\001\030\001\031\001\032\001\033\001\034\001\035\001\
\036\001\037\001\038\001\039\001\016\001\039\000\094\000\019\001\
\020\001\021\001\086\000\023\001\024\001\025\001\026\001\027\001\
\028\001\029\001\030\001\031\001\032\001\033\001\034\001\035\001\
\036\001\037\001\038\001\039\001\089\000\174\000\036\000\048\000\
\051\000\176\000\107\000\172\000\198\000"

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
  Tswitchless\000\
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
# 332 "Parser.mly"
                 ( Ast.Char Ast.Signed )
# 704 "Parser.ml"
               : 'char_type))
; (fun __caml_parser_env ->
    Obj.repr(
# 333 "Parser.mly"
                    ( Ast.Char Ast.Unsigned )
# 710 "Parser.ml"
               : 'char_type))
; (fun __caml_parser_env ->
    Obj.repr(
# 337 "Parser.mly"
                     ( Ast.IShort )
# 716 "Parser.ml"
               : 'ex_shortness))
; (fun __caml_parser_env ->
    Obj.repr(
# 338 "Parser.mly"
          ( Ast.ILong )
# 722 "Parser.ml"
               : 'ex_shortness))
; (fun __caml_parser_env ->
    Obj.repr(
# 341 "Parser.mly"
                          ( Ast.LLong Ast.Signed )
# 728 "Parser.ml"
               : 'longlong))
; (fun __caml_parser_env ->
    Obj.repr(
# 342 "Parser.mly"
                          ( Ast.LLong Ast.Unsigned )
# 734 "Parser.ml"
               : 'longlong))
; (fun __caml_parser_env ->
    Obj.repr(
# 344 "Parser.mly"
                       ( Ast.INone )
# 740 "Parser.ml"
               : 'shortness))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'ex_shortness) in
    Obj.repr(
# 345 "Parser.mly"
                 ( _1 )
# 747 "Parser.ml"
               : 'shortness))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'shortness) in
    Obj.repr(
# 348 "Parser.mly"
                         (
      Ast.Int { Ast.ia_signedness = Ast.Signed; Ast.ia_shortness = _1 }
    )
# 756 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'shortness) in
    Obj.repr(
# 351 "Parser.mly"
                             (
      Ast.Int { Ast.ia_signedness = Ast.Unsigned; Ast.ia_shortness = _2 }
    )
# 765 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'shortness) in
    Obj.repr(
# 354 "Parser.mly"
                        (
      Ast.Int { Ast.ia_signedness = Ast.Unsigned; Ast.ia_shortness = _2 }
    )
# 774 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'longlong) in
    Obj.repr(
# 357 "Parser.mly"
             ( _1 )
# 781 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'ex_shortness) in
    Obj.repr(
# 358 "Parser.mly"
                 (
      Ast.Int { Ast.ia_signedness = Ast.Signed; Ast.ia_shortness = _1 }
    )
# 790 "Parser.ml"
               : 'int_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'char_type) in
    Obj.repr(
# 364 "Parser.mly"
              ( _1 )
# 797 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'int_type) in
    Obj.repr(
# 365 "Parser.mly"
              ( _1 )
# 804 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 367 "Parser.mly"
             ( Ast.Float )
# 810 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 368 "Parser.mly"
             ( Ast.Double )
# 816 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 369 "Parser.mly"
                  ( Ast.LDouble )
# 822 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 371 "Parser.mly"
             ( Ast.Int8 )
# 828 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 372 "Parser.mly"
             ( Ast.Int16 )
# 834 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 373 "Parser.mly"
             ( Ast.Int32 )
# 840 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 374 "Parser.mly"
             ( Ast.Int64 )
# 846 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 375 "Parser.mly"
             ( Ast.UInt8 )
# 852 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 376 "Parser.mly"
             ( Ast.UInt16 )
# 858 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 377 "Parser.mly"
             ( Ast.UInt32 )
# 864 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 378 "Parser.mly"
             ( Ast.UInt64 )
# 870 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 379 "Parser.mly"
             ( Ast.SizeT )
# 876 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 380 "Parser.mly"
             ( Ast.WChar )
# 882 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 381 "Parser.mly"
             ( Ast.Void )
# 888 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'struct_specifier) in
    Obj.repr(
# 383 "Parser.mly"
                     ( _1 )
# 895 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'union_specifier) in
    Obj.repr(
# 384 "Parser.mly"
                     ( _1 )
# 902 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'enum_specifier) in
    Obj.repr(
# 385 "Parser.mly"
                     ( _1 )
# 909 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 386 "Parser.mly"
                     ( Ast.Foreign(_1) )
# 916 "Parser.ml"
               : 'type_spec))
; (fun __caml_parser_env ->
    Obj.repr(
# 389 "Parser.mly"
                 ( fun ii -> Ast.Ptr(ii) )
# 922 "Parser.ml"
               : 'pointer))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'pointer) in
    Obj.repr(
# 390 "Parser.mly"
                 ( fun ii -> Ast.Ptr(_1 ii) )
# 929 "Parser.ml"
               : 'pointer))
; (fun __caml_parser_env ->
    Obj.repr(
# 393 "Parser.mly"
                                         ( failwith "Flexible array is not supported." )
# 935 "Parser.ml"
               : 'empty_dimension))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : int) in
    Obj.repr(
# 394 "Parser.mly"
                                         ( if _2 <> 0 then [_2]
                                           else failwith "Zero-length array is not supported." )
# 943 "Parser.ml"
               : 'fixed_dimension))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'fixed_dimension) in
    Obj.repr(
# 397 "Parser.mly"
                                     ( _1 )
# 950 "Parser.ml"
               : 'fixed_size_array))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'fixed_size_array) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'fixed_dimension) in
    Obj.repr(
# 398 "Parser.mly"
                                     ( _1 @ _2 )
# 958 "Parser.ml"
               : 'fixed_size_array))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'fixed_size_array) in
    Obj.repr(
# 401 "Parser.mly"
                                     ( _1 )
# 965 "Parser.ml"
               : 'array_size))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'empty_dimension) in
    Obj.repr(
# 402 "Parser.mly"
                                     ( _1 )
# 972 "Parser.ml"
               : 'array_size))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'empty_dimension) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'fixed_size_array) in
    Obj.repr(
# 403 "Parser.mly"
                                     ( _1 @ _2 )
# 980 "Parser.ml"
               : 'array_size))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'type_spec) in
    Obj.repr(
# 406 "Parser.mly"
                      ( _1 )
# 987 "Parser.ml"
               : 'all_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'type_spec) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'pointer) in
    Obj.repr(
# 407 "Parser.mly"
                      ( _2 _1 )
# 995 "Parser.ml"
               : 'all_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 410 "Parser.mly"
                           ( { Ast.identifier = _1; Ast.array_dims = []; } )
# 1002 "Parser.ml"
               : 'declarator))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'array_size) in
    Obj.repr(
# 411 "Parser.mly"
                           ( { Ast.identifier = _1; Ast.array_dims = _2; } )
# 1010 "Parser.ml"
               : 'declarator))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'attr_block) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'all_type) in
    Obj.repr(
# 420 "Parser.mly"
                                (
    let attr = get_param_ptr_attr _1 in
    (*check the type is build in type or used defined type.*)
    let rec is_foreign s =
      match s with
        Ast.Ptr(a) -> is_foreign a
      | Ast.Foreign _ -> true
      | _ -> false
    in
    let is_bare_foreign s =
      match s with
      | Ast.Foreign _ -> true
      | _ -> false
    in
    (*'isptr', 'isary', only allowed for bare user defined type.*)
    (*'readonly' only allowed for user defined type.*)
    if attr.Ast.pa_isptr && not (is_bare_foreign _2) then
      failwithf "'isptr', attributes are only for user defined type, not for `%s'." (Ast.get_tystr _2)
    else if attr.Ast.pa_isary && not (is_bare_foreign _2) then
      failwithf "'isary', attributes are only for user defined type, not for `%s'." (Ast.get_tystr _2)
    else if attr.Ast.pa_rdonly && not (is_foreign _2) then
      failwithf "'readonly', attributes are only for user defined type, not for `%s'." (Ast.get_tystr _2)
    else if attr.Ast.pa_rdonly && not (attr.Ast.pa_isptr) then
      failwithf "'readonly' attribute is only used with 'isptr' attribute."    else
    match _2 with
      Ast.Ptr _ -> fun x -> Ast.PTPtr(_2, get_param_ptr_attr _1)
    | _         ->
      if _1 <> [] then
        match _2 with
          Ast.Foreign s ->
            if attr.Ast.pa_isptr || attr.Ast.pa_isary then fun x -> Ast.PTPtr(_2, attr)
            else
              (* thinking about 'user_defined_type var[4]' *)
              fun is_ary ->
                if is_ary then Ast.PTPtr(_2, attr)
                else failwithf "`%s' is considered plain type but decorated with pointer attributes" s
        | _ ->
          fun is_ary ->
            if is_ary then Ast.PTPtr(_2, attr)
            else failwithf "unexpected pointer attributes for `%s'" (Ast.get_tystr _2)
      else
        fun is_ary ->
          if is_ary then Ast.PTPtr(_2, get_param_ptr_attr [])
          else  Ast.PTVal _2
    )
# 1062 "Parser.ml"
               : 'param_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'all_type) in
    Obj.repr(
# 465 "Parser.mly"
             (
    match _1 with
      Ast.Ptr _ -> fun x -> Ast.PTPtr(_1, get_param_ptr_attr [])
    | _         ->
      fun is_ary ->
        if is_ary then Ast.PTPtr(_1, get_param_ptr_attr [])
        else  Ast.PTVal _1
    )
# 1076 "Parser.ml"
               : 'param_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'attr_block) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'type_spec) in
    let _4 = (Parsing.peek_val __caml_parser_env 0 : 'pointer) in
    Obj.repr(
# 473 "Parser.mly"
                                        (
      let attr = get_param_ptr_attr _1
      in fun x -> Ast.PTPtr(_4 _3, { attr with Ast.pa_rdonly = true })
    )
# 1088 "Parser.ml"
               : 'param_type))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'type_spec) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'pointer) in
    Obj.repr(
# 477 "Parser.mly"
                             (
      let attr = get_param_ptr_attr []
      in fun x -> Ast.PTPtr(_3 _2, { attr with Ast.pa_rdonly = true })
    )
# 1099 "Parser.ml"
               : 'param_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'attr_block) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'all_type) in
    Obj.repr(
# 490 "Parser.mly"
                                  (
    let attr = get_member_ptr_attr _1 in
    (*'isptr', 'isary', 'readonly' not allowed.*)
    if attr.Ast.pa_direction <> Ast.PtrNoDirection then
      failwithf "direction attribute is not allowed for structure member."
    else if attr.Ast.pa_isptr then
      failwithf "'isptr', attribute is not allowed for structure member."
    else if attr.Ast.pa_isary then
      failwithf "'isary', attribute is not allowed for structure member."
    else if attr.Ast.pa_rdonly then
      failwithf "'readonly'attribute is not allowed for structure member."
    else if attr.Ast.pa_isstr then
      failwithf "'string'attribute is not allowed for structure member."
    else if attr.Ast.pa_iswstr then
      failwithf "'wstring' attribute is not allowed for structure member."
    else
    match _2 with
      Ast.Ptr _ -> fun x -> Ast.PTPtr(_2, get_member_ptr_attr _1)
    | _         ->
      if _1 <> [] then
        match _2 with
          Ast.Foreign s ->
              (* thinking about 'user_defined_type var[4]' *)
              fun is_ary ->
                if is_ary then Ast.PTPtr(_2, attr)
                else failwithf "`%s' is considered plain type but decorated with pointer attributes" s
        | _ ->
          fun is_ary ->
            if is_ary then Ast.PTPtr(_2, attr)
            else failwithf "'%s' specified deep copy, but it's not a pointer" (Ast.get_tystr _2)
      else
        fun is_ary ->
          if is_ary then Ast.PTPtr(_2, get_ptr_attr [])
          else Ast.PTVal _2
    )
# 1141 "Parser.ml"
               : 'smember_type))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'all_type) in
    Obj.repr(
# 525 "Parser.mly"
             (
    match _1 with
      Ast.Ptr _ -> fun x -> Ast.PTPtr(_1, get_ptr_attr [])
    | _         ->
      fun is_ary ->
        if is_ary then Ast.PTPtr(_1, get_ptr_attr [])
        else  Ast.PTVal _1
  )
# 1155 "Parser.ml"
               : 'smember_type))
; (fun __caml_parser_env ->
    Obj.repr(
# 535 "Parser.mly"
                                  ( failwith "no attribute specified." )
# 1161 "Parser.ml"
               : 'attr_block))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'key_val_pairs) in
    Obj.repr(
# 536 "Parser.mly"
                                  ( _2 )
# 1168 "Parser.ml"
               : 'attr_block))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'key_val_pair) in
    Obj.repr(
# 539 "Parser.mly"
                                      ( [_1] )
# 1175 "Parser.ml"
               : 'key_val_pairs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'key_val_pairs) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'key_val_pair) in
    Obj.repr(
# 540 "Parser.mly"
                                      (  _3 :: _1 )
# 1183 "Parser.ml"
               : 'key_val_pairs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 543 "Parser.mly"
                                             ( (_1, Ast.AString(_3)) )
# 1191 "Parser.ml"
               : 'key_val_pair))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : int) in
    Obj.repr(
# 544 "Parser.mly"
                                             ( (_1, Ast.ANumber(_3)) )
# 1199 "Parser.ml"
               : 'key_val_pair))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 545 "Parser.mly"
                                             ( (_1, Ast.AString("")) )
# 1206 "Parser.ml"
               : 'key_val_pair))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 548 "Parser.mly"
                                      ( Ast.Struct(_2) )
# 1213 "Parser.ml"
               : 'struct_specifier))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 549 "Parser.mly"
                                      ( Ast.Union(_2) )
# 1220 "Parser.ml"
               : 'union_specifier))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 550 "Parser.mly"
                                      ( Ast.Enum(_2) )
# 1227 "Parser.ml"
               : 'enum_specifier))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'struct_specifier) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'struct_member_list) in
    Obj.repr(
# 552 "Parser.mly"
                                                                       (
    let s = { Ast.sname = (match _1 with Ast.Struct s -> s | _ -> "");
              Ast.smlist = List.rev _3; }
    in Ast.StructDef(s)
  )
# 1239 "Parser.ml"
               : 'struct_definition))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'union_specifier) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'union_member_list) in
    Obj.repr(
# 558 "Parser.mly"
                                                                    (
    let s = { Ast.uname = (match _1 with Ast.Union s -> s | _ -> "");
              Ast.umlist = List.rev _3; }
    in Ast.UnionDef(s)
  )
# 1251 "Parser.ml"
               : 'union_definition))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'enum_body) in
    Obj.repr(
# 565 "Parser.mly"
                                                 (
      let e = { Ast.enname = ""; Ast.enbody = _3; }
      in Ast.EnumDef(e)
    )
# 1261 "Parser.ml"
               : 'enum_definition))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'enum_specifier) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'enum_body) in
    Obj.repr(
# 569 "Parser.mly"
                                             (
      let e = { Ast.enname = (match _1 with Ast.Enum s -> s | _ -> "");
                Ast.enbody = _3; }
      in Ast.EnumDef(e)
    )
# 1273 "Parser.ml"
               : 'enum_definition))
; (fun __caml_parser_env ->
    Obj.repr(
# 576 "Parser.mly"
                       ( [] )
# 1279 "Parser.ml"
               : 'enum_body))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'enum_eles) in
    Obj.repr(
# 577 "Parser.mly"
                       ( List.rev _1 )
# 1286 "Parser.ml"
               : 'enum_body))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'enum_ele) in
    Obj.repr(
# 580 "Parser.mly"
                              ( [_1] )
# 1293 "Parser.ml"
               : 'enum_eles))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'enum_eles) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'enum_ele) in
    Obj.repr(
# 581 "Parser.mly"
                              ( _3 :: _1 )
# 1301 "Parser.ml"
               : 'enum_eles))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 584 "Parser.mly"
                                   ( (_1, Ast.EnumValNone) )
# 1308 "Parser.ml"
               : 'enum_ele))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 585 "Parser.mly"
                                   ( (_1, Ast.EnumVal (Ast.AString _3)) )
# 1316 "Parser.ml"
               : 'enum_ele))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : int) in
    Obj.repr(
# 586 "Parser.mly"
                                   ( (_1, Ast.EnumVal (Ast.ANumber _3)) )
# 1324 "Parser.ml"
               : 'enum_ele))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'struct_definition) in
    Obj.repr(
# 589 "Parser.mly"
                                      ( _1 )
# 1331 "Parser.ml"
               : 'composite_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'union_definition) in
    Obj.repr(
# 590 "Parser.mly"
                                      ( _1 )
# 1338 "Parser.ml"
               : 'composite_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'enum_definition) in
    Obj.repr(
# 591 "Parser.mly"
                                      ( _1 )
# 1345 "Parser.ml"
               : 'composite_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'struct_member_def) in
    Obj.repr(
# 594 "Parser.mly"
                                                    ( [_1] )
# 1352 "Parser.ml"
               : 'struct_member_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'struct_member_list) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'struct_member_def) in
    Obj.repr(
# 595 "Parser.mly"
                                                    ( _2 :: _1 )
# 1360 "Parser.ml"
               : 'struct_member_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'union_member_def) in
    Obj.repr(
# 598 "Parser.mly"
                                                  ( [_1] )
# 1367 "Parser.ml"
               : 'union_member_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'union_member_list) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'union_member_def) in
    Obj.repr(
# 599 "Parser.mly"
                                                  ( _2 :: _1 )
# 1375 "Parser.ml"
               : 'union_member_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'smember_type) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'declarator) in
    Obj.repr(
# 602 "Parser.mly"
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
# 1394 "Parser.ml"
               : 'struct_member_def))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'all_type) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'declarator) in
    Obj.repr(
# 615 "Parser.mly"
                                      (
    if _1 = Ast.Void then
      failwithf "union member `%s' has `void' type." _2.Ast.identifier
    else
      (_1, _2)
)
# 1407 "Parser.ml"
               : 'union_member_def))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 625 "Parser.mly"
                                  ( [_1] )
# 1414 "Parser.ml"
               : 'func_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'func_list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 626 "Parser.mly"
                                  ( _3 :: _1 )
# 1422 "Parser.ml"
               : 'func_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 629 "Parser.mly"
                                  ( _1 )
# 1429 "Parser.ml"
               : 'module_path))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 2 : 'module_path) in
    let _4 = (Parsing.peek_val __caml_parser_env 0 : 'func_list) in
    Obj.repr(
# 631 "Parser.mly"
                                                         (
      { Ast.mname = _2; Ast.flist = List.rev _4; }
    )
# 1439 "Parser.ml"
               : 'import_declaration))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 2 : 'module_path) in
    Obj.repr(
# 634 "Parser.mly"
                                   (
      { Ast.mname = _2; Ast.flist = ["*"]; }
    )
# 1448 "Parser.ml"
               : 'import_declaration))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 639 "Parser.mly"
                                      ( _2 )
# 1455 "Parser.ml"
               : 'include_declaration))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'include_declaration) in
    Obj.repr(
# 641 "Parser.mly"
                                             ( [_1] )
# 1462 "Parser.ml"
               : 'include_declarations))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'include_declarations) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'include_declaration) in
    Obj.repr(
# 642 "Parser.mly"
                                             ( _2 :: _1 )
# 1470 "Parser.ml"
               : 'include_declarations))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 2 : 'trusted_block) in
    Obj.repr(
# 648 "Parser.mly"
                                                                     (
      List.rev _3
    )
# 1479 "Parser.ml"
               : 'enclave_functions))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 2 : 'untrusted_block) in
    Obj.repr(
# 651 "Parser.mly"
                                                          (
      List.rev _3
    )
# 1488 "Parser.ml"
               : 'enclave_functions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'trusted_functions) in
    Obj.repr(
# 656 "Parser.mly"
                                             ( _1 )
# 1495 "Parser.ml"
               : 'trusted_block))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'include_declarations) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'trusted_functions) in
    Obj.repr(
# 657 "Parser.mly"
                                             (
      trusted_headers := !trusted_headers @ List.rev _1; _2
    )
# 1505 "Parser.ml"
               : 'trusted_block))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'untrusted_functions) in
    Obj.repr(
# 662 "Parser.mly"
                                             ( _1 )
# 1512 "Parser.ml"
               : 'untrusted_block))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'include_declarations) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'untrusted_functions) in
    Obj.repr(
# 663 "Parser.mly"
                                             (
      untrusted_headers := !untrusted_headers @ List.rev _1; _2
    )
# 1522 "Parser.ml"
               : 'untrusted_block))
; (fun __caml_parser_env ->
    Obj.repr(
# 669 "Parser.mly"
                               ( true )
# 1528 "Parser.ml"
               : 'access_modifier))
; (fun __caml_parser_env ->
    Obj.repr(
# 670 "Parser.mly"
                               ( false  )
# 1534 "Parser.ml"
               : 'access_modifier))
; (fun __caml_parser_env ->
    Obj.repr(
# 674 "Parser.mly"
                                     ( false )
# 1540 "Parser.ml"
               : 'switchless_annotation))
; (fun __caml_parser_env ->
    Obj.repr(
# 675 "Parser.mly"
                                     ( true  )
# 1546 "Parser.ml"
               : 'switchless_annotation))
; (fun __caml_parser_env ->
    Obj.repr(
# 678 "Parser.mly"
                                          ( [] )
# 1552 "Parser.ml"
               : 'trusted_functions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 4 : 'trusted_functions) in
    let _2 = (Parsing.peek_val __caml_parser_env 3 : 'access_modifier) in
    let _3 = (Parsing.peek_val __caml_parser_env 2 : 'func_def) in
    let _4 = (Parsing.peek_val __caml_parser_env 1 : 'switchless_annotation) in
    Obj.repr(
# 679 "Parser.mly"
                                                                                (
      check_ptr_attr _3 (symbol_start_pos(), symbol_end_pos());
      Ast.Trusted { Ast.tf_fdecl = _3; Ast.tf_is_priv = _2; Ast.tf_is_switchless = _4 } :: _1
    )
# 1565 "Parser.ml"
               : 'trusted_functions))
; (fun __caml_parser_env ->
    Obj.repr(
# 685 "Parser.mly"
                                                      ( [] )
# 1571 "Parser.ml"
               : 'untrusted_functions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'untrusted_functions) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'untrusted_func_def) in
    Obj.repr(
# 686 "Parser.mly"
                                                      ( _2 :: _1 )
# 1579 "Parser.ml"
               : 'untrusted_functions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'all_type) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'parameter_list) in
    Obj.repr(
# 689 "Parser.mly"
                                              (
      { Ast.fname = _2; Ast.rtype = _1; Ast.plist = List.rev _3 ; }
    )
# 1590 "Parser.ml"
               : 'func_def))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'all_type) in
    let _2 = (Parsing.peek_val __caml_parser_env 2 : 'array_size) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : string) in
    let _4 = (Parsing.peek_val __caml_parser_env 0 : 'parameter_list) in
    Obj.repr(
# 692 "Parser.mly"
                                                   (
      failwithf "%s: returning an array is not supported - use pointer instead." _3
    )
# 1602 "Parser.ml"
               : 'func_def))
; (fun __caml_parser_env ->
    Obj.repr(
# 697 "Parser.mly"
                                   ( [] )
# 1608 "Parser.ml"
               : 'parameter_list))
; (fun __caml_parser_env ->
    Obj.repr(
# 698 "Parser.mly"
                                   ( [] )
# 1614 "Parser.ml"
               : 'parameter_list))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'parameter_defs) in
    Obj.repr(
# 699 "Parser.mly"
                                   ( _2 )
# 1621 "Parser.ml"
               : 'parameter_list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'parameter_def) in
    Obj.repr(
# 702 "Parser.mly"
                                        ( [_1] )
# 1628 "Parser.ml"
               : 'parameter_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'parameter_defs) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'parameter_def) in
    Obj.repr(
# 703 "Parser.mly"
                                        ( _3 :: _1 )
# 1636 "Parser.ml"
               : 'parameter_defs))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'param_type) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'declarator) in
    Obj.repr(
# 706 "Parser.mly"
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
# 1655 "Parser.ml"
               : 'parameter_def))
; (fun __caml_parser_env ->
    Obj.repr(
# 720 "Parser.mly"
                               ( false )
# 1661 "Parser.ml"
               : 'propagate_errno))
; (fun __caml_parser_env ->
    Obj.repr(
# 721 "Parser.mly"
                               ( true  )
# 1667 "Parser.ml"
               : 'propagate_errno))
; (fun __caml_parser_env ->
    Obj.repr(
# 724 "Parser.mly"
                                  ( [] )
# 1673 "Parser.ml"
               : 'untrusted_prefixes))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'attr_block) in
    Obj.repr(
# 725 "Parser.mly"
                         ( _1  )
# 1680 "Parser.ml"
               : 'untrusted_prefixes))
; (fun __caml_parser_env ->
    Obj.repr(
# 728 "Parser.mly"
                                     (  (false, false) )
# 1686 "Parser.ml"
               : 'untrusted_postfixes))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'switchless_annotation) in
    Obj.repr(
# 729 "Parser.mly"
                                            ( (true, _2) )
# 1693 "Parser.ml"
               : 'untrusted_postfixes))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'propagate_errno) in
    Obj.repr(
# 730 "Parser.mly"
                                 ( (_2, true) )
# 1700 "Parser.ml"
               : 'untrusted_postfixes))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : 'untrusted_prefixes) in
    let _2 = (Parsing.peek_val __caml_parser_env 2 : 'func_def) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'allow_list) in
    let _4 = (Parsing.peek_val __caml_parser_env 0 : 'untrusted_postfixes) in
    Obj.repr(
# 733 "Parser.mly"
                                                                               (
      check_ptr_attr _2 (symbol_start_pos(), symbol_end_pos());
      let fattr = get_func_attr _1 in
      Ast.Untrusted { Ast.uf_fdecl = _2; Ast.uf_fattr = fattr; Ast.uf_allow_list = _3; Ast.uf_propagate_errno = fst _4; Ast.uf_is_switchless = snd _4; }
    )
# 1714 "Parser.ml"
               : 'untrusted_func_def))
; (fun __caml_parser_env ->
    Obj.repr(
# 740 "Parser.mly"
                                     ( [] )
# 1720 "Parser.ml"
               : 'allow_list))
; (fun __caml_parser_env ->
    Obj.repr(
# 741 "Parser.mly"
                                     ( [] )
# 1726 "Parser.ml"
               : 'allow_list))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'func_list) in
    Obj.repr(
# 742 "Parser.mly"
                                     ( _3 )
# 1733 "Parser.ml"
               : 'allow_list))
; (fun __caml_parser_env ->
    Obj.repr(
# 748 "Parser.mly"
                           ( [] )
# 1739 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'expressions) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'include_declaration) in
    Obj.repr(
# 749 "Parser.mly"
                                              ( Ast.Include(_2)   :: _1 )
# 1747 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'expressions) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'import_declaration) in
    Obj.repr(
# 750 "Parser.mly"
                                              ( Ast.Importing(_2) :: _1 )
# 1755 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'expressions) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : 'composite_defs) in
    Obj.repr(
# 751 "Parser.mly"
                                              ( Ast.Composite(_2) :: _1 )
# 1763 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'expressions) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : 'enclave_functions) in
    Obj.repr(
# 752 "Parser.mly"
                                              ( Ast.Interface(_2) :: _1 )
# 1771 "Parser.ml"
               : 'expressions))
; (fun __caml_parser_env ->
    let _3 = (Parsing.peek_val __caml_parser_env 1 : 'expressions) in
    Obj.repr(
# 755 "Parser.mly"
                                                  (
      { Ast.ename = "";
        Ast.eexpr = List.rev _3 }
    )
# 1781 "Parser.ml"
               : 'enclave_def))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'enclave_def) in
    Obj.repr(
# 764 "Parser.mly"
                                          ( _1 )
# 1788 "Parser.ml"
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
