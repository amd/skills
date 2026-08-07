#!/usr/bin/env python3
"""
Validate a generated collection.router policy JSON against the same structural
and numeric rules the Lemonade server's C++ parser (routing_policy_parser.cpp)
enforces at POST /v1/pull - offline, no server or network required.

Catches the class of mistake easy to make when hand-authoring nested JSON: an
out-of-range score, a negative char count, an unbalanced match expression, a
default_label that isn't in labels, a semantic_similarity classifier that
declares labels, an llm classifier missing labels, etc.

This does NOT (and cannot, without a live server) check whether a named model
actually exists or has the right capability (chat/embedding/classification) -
that needs GET /api/v1/models/{id} against a running lemond; see SKILL.md
Step 8 for that optional live check.

Usage:
    python3 scripts/validate.py router.json
    python3 scripts/validate.py < router.json
    cat router.json | python3 scripts/validate.py -

Exits 0 if no errors (warnings/advisories may still be present), 1 otherwise.
JSON to stdout.
"""

import argparse
import json
import re
import sys

LEAF_KEYS = {
    "keywords_any", "keywords_all", "regex", "min_chars", "max_chars",
    "has_tools", "has_images", "classifier", "label", "min_score",
    "max_score", "metadata",
}
LOGICAL_KEYS = {"all", "any", "not"}
CLASSIFIER_TYPES = {"classifier", "semantic_similarity", "llm"}
ON_ERROR_VALUES = {"match_false", "match_true"}
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

ROOT_KEYS = {"version", "model_name", "recipe", "components", "models", "routing"}
ROUTING_KEYS = {"candidates", "default_model", "router", "rules", "classifiers"}
CLASSIFIER_KEYS = {"id", "type", "model", "labels", "default_label", "on_error",
                   "prompt", "reference_phrases"}
BAD_PROMPT_RE = re.compile(
    r"reply with only the exact model name|only the model name|"
    r"reply with only.{0,20}model name|"
    r"reply with (exactly )?one label|"
    r"reply with only.{0,20}label|"
    r"output only.{0,20}label|"
    r"respond with only.{0,20}label|"
    r"^\s*pick\s+\S|"                  # "Pick <model>" imperative at start of prompt
    r"respond with.{0,20}model name|"
    r"output.{0,20}model name", re.I | re.MULTILINE)
# Keep the old name as an alias so existing code that references it still works
BAD_ROUTER_PROMPT_RE = BAD_PROMPT_RE

# Nested unbounded quantifiers rejected by std::regex (ECMAScript) as ReDoS safeguard.
# Python re compiles these fine, so we catch them with a simple structural check.
_REDOS_RE = re.compile(r"\([^)]*[+*][^)]*\)[+*]|[+*]\{[^}]*\}[+*]")


# Constructs accepted by Node/V8 (ES2018+) but rejected by std::regex ECMAScript
# (the C++ engine the server uses, which implements ~ES5.1).
# This scan runs before Node so that Node's newer engine can't give a false pass.
_STDREGEX_UNSUPPORTED_RE = re.compile(
    r"\(\?P[<']"           # Python named group (?P<name> / (?P'name' — not ECMAScript at all
    r"|\(\?P=[^)]*\)"      # Python backreference (?P=name)
    r"|\(\?#[^)]*\)"       # Python comment (?#...)
    r"|\(\?<=[^)]*\)"      # lookbehind (?<=...) — ES2018, not in std::regex
    r"|\(\?<![^)]*\)"      # negative lookbehind (?<!...) — ES2018, not in std::regex
    r"|\(\?<[A-Za-z_]"     # ECMAScript named group (?<name>...) — ES2018, not in std::regex
    r"|\\[pP]\{"           # Unicode property escape \p{...} / \P{...} — ES2018, not in std::regex
)
# Keep the old name so any external code that references it doesn't break.
_PYTHON_ONLY_RE = _STDREGEX_UNSUPPORTED_RE


def _check_ecmascript_regex(pattern):
    """Return (error_str, skip_notice) for pattern against ECMAScript regex rules.

    Always runs the dialect scan first: rejects constructs that are valid in
    Python re (and may be accepted by Node v8+) but are not supported by
    std::regex with the ECMAScript flag (the C++ engine the server uses).
    Lookbehinds ((?<=...) / (?<!...)) are the canonical example — accepted by
    Node since ES2018 but rejected by std::regex on all supported platforms.

    Then uses Node.js (new RegExp) when available for a second-pass syntax
    check. Falls back to Python re when Node is absent, with a warning.
    """
    import shutil
    import subprocess

    # Dialect check runs unconditionally — before Node — because Node's engine
    # (V8, ES2018+) accepts features that std::regex ECMAScript (~ES5.1) does not:
    # lookbehinds, named groups (?<name>…), Unicode property escapes \p{…}, etc.
    m = _STDREGEX_UNSUPPORTED_RE.search(pattern)
    if m:
        return (
            f"regex construct '{m.group()}' is not supported by std::regex ECMAScript "
            f"(the server's C++ engine, ~ES5.1); Node/V8 accepts this but the server will reject it",
            None,
        )

    node = shutil.which("node") or shutil.which("nodejs")
    if not node:
        # Best-effort syntax check via Python re.
        try:
            re.compile(pattern)
        except re.error as exc:
            return f"regex syntax error: {exc}", None
        return None, "node/nodejs not found - ECMAScript regex validation skipped; install Node.js to enable"

    # Wrap in try/catch so Node exits 0 on valid, 1 with the SyntaxError message on invalid.
    script = (
        "try { new RegExp(" + json.dumps(pattern) + "); process.exit(0); } "
        "catch(e) { process.stdout.write(e.message); process.exit(1); }"
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        return result.stdout.strip() or "invalid ECMAScript regex", None
    return None, None


def _err(issues, check, message, path=""):
    issues.append({"check": check, "severity": "error",
                   "message": f"{path}: {message}" if path else message})


def _warn(issues, check, message, path=""):
    issues.append({"check": check, "severity": "warning",
                   "message": f"{path}: {message}" if path else message})


def _is_non_negative_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _is_score(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0


def validate_match_expr(expr, classifiers, path, issues, depth=0):
    if depth > 64:
        _err(issues, "match_depth", "exceeds max nesting depth (64)", path)
        return
    if not isinstance(expr, dict) or not expr:
        _err(issues, "match_shape", "must be a non-empty object", path)
        return

    logical = LOGICAL_KEYS & expr.keys()
    if logical:
        if len(expr) != 1:
            _err(issues, "match_shape",
                 f"logical key ({', '.join(sorted(logical))}) cannot be mixed with other keys", path)
            return
        key = next(iter(logical))
        if key == "not":
            validate_match_expr(expr["not"], classifiers, f"{path}.not", issues, depth + 1)
        else:
            children = expr[key]
            if not isinstance(children, list) or not children:
                _err(issues, "match_shape", f"'{key}' must be a non-empty array", path)
                return
            for i, child in enumerate(children):
                validate_match_expr(child, classifiers, f"{path}.{key}[{i}]", issues, depth + 1)
        return

    # Leaf node
    unknown = set(expr.keys()) - LEAF_KEYS
    if unknown:
        _err(issues, "unknown_key", f"unknown condition key(s): {', '.join(sorted(unknown))}", path)

    condition_count = 0
    for key in ("keywords_any", "keywords_all"):
        if key not in expr:
            continue
        condition_count += 1
        v = expr[key]
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x for x in v):
            _err(issues, "keywords", f"'{key}' must be a non-empty array of non-empty strings", path)

    if "regex" in expr:
        condition_count += 1
        v = expr["regex"]
        if not isinstance(v, str) or not v:
            _err(issues, "regex", "'regex' must be a non-empty string", path)
        else:
            ecma_err, ecma_skip = _check_ecmascript_regex(v)
            if ecma_err:
                _err(issues, "regex_dialect",
                     f"invalid ECMAScript regex: {ecma_err}", path)
            elif ecma_skip:
                _warn(issues, "regex_dialect", ecma_skip, path)
            if _REDOS_RE.search(v):
                _err(issues, "regex_redos",
                     "pattern contains nested unbounded quantifiers (e.g. (X+)+) which the "
                     "server rejects at load time as a ReDoS safeguard", path)

    for key in ("min_chars", "max_chars"):
        if key in expr:
            condition_count += 1
            if not _is_non_negative_int(expr[key]):
                _err(issues, "char_count", f"'{key}' must be a non-negative integer, got {expr[key]!r}", path)

    for key in ("has_tools", "has_images"):
        if key in expr:
            condition_count += 1
            if not isinstance(expr[key], bool):
                _err(issues, "boolean_condition", f"'{key}' must be a boolean, got {expr[key]!r}", path)

    if "classifier" in expr:
        condition_count += 1
        cid = expr["classifier"]
        if not isinstance(cid, str) or not cid:
            _err(issues, "classifier_ref", "'classifier' must be a non-empty string", path)
        elif cid not in classifiers:
            _err(issues, "classifier_ref", f"references undeclared classifier '{cid}'", path)
        else:
            clf = classifiers[cid]
            pool = (list((clf.get("reference_phrases") or {}).keys())
                    if clf.get("type") == "semantic_similarity" else (clf.get("labels") or []))
            label = expr.get("label")
            if label is not None:
                if not isinstance(label, str) or not label:
                    _err(issues, "classifier_label", "'label' must be a non-empty string", path)
                elif pool and label not in pool:
                    _err(issues, "classifier_label", f"label '{label}' is not declared on classifier '{cid}'", path)
            elif pool and not clf.get("default_label"):
                _err(issues, "classifier_label",
                     f"omits 'label' but classifier '{cid}' has labels and no default_label", path)

        min_score = expr.get("min_score")
        max_score = expr.get("max_score")
        if min_score is not None and not _is_score(min_score):
            _err(issues, "score_range", f"'min_score' must be in [0, 1], got {min_score!r}", path)
        if max_score is not None and not _is_score(max_score):
            _err(issues, "score_range", f"'max_score' must be in [0, 1], got {max_score!r}", path)
        if (_is_score(min_score) if min_score is not None else False) and \
           (_is_score(max_score) if max_score is not None else False) and min_score > max_score:
            _err(issues, "score_range", f"min_score ({min_score}) exceeds max_score ({max_score})", path)
    elif "label" in expr or "min_score" in expr or "max_score" in expr:
        _err(issues, "classifier_ref", "'label'/'min_score'/'max_score' require a 'classifier' key in the same leaf", path)

    if "metadata" in expr:
        condition_count += 1
        md = expr["metadata"]
        if not isinstance(md, dict) or not isinstance(md.get("key"), str) or not md.get("key"):
            _err(issues, "metadata", "requires a non-empty string 'key'", path)
        else:
            unknown_md = set(md.keys()) - {"key", "equals", "any", "exists"}
            if unknown_md:
                _err(issues, "metadata", f"unknown key(s): {', '.join(sorted(unknown_md))}", path)
            comparators = [c for c in ("equals", "any", "exists") if c in md]
            if len(comparators) != 1:
                _err(issues, "metadata", "must have exactly one of equals/any/exists", path)
            elif comparators[0] == "equals" and not isinstance(md["equals"], str):
                _err(issues, "metadata", "'equals' must be a string", path)
            elif comparators[0] == "any" and (
                    not isinstance(md["any"], list) or not md["any"]
                    or not all(isinstance(x, str) and x for x in md["any"])):
                _err(issues, "metadata", "'any' must be a non-empty array of non-empty strings", path)
            elif comparators[0] == "exists" and not isinstance(md["exists"], bool):
                _err(issues, "metadata", "'exists' must be a boolean", path)

    if condition_count == 0:
        _err(issues, "empty_leaf", "leaf has no recognized condition", path)


def validate(policy):
    issues = []

    if not isinstance(policy, dict):
        _err(issues, "root", "policy must be a JSON object")
        return issues

    unknown_root = set(policy.keys()) - ROOT_KEYS
    if unknown_root:
        _err(issues, "unknown_key", f"unknown root key(s): {', '.join(sorted(unknown_root))}")

    if policy.get("version") != "1":
        _err(issues, "version", f"must be the string \"1\", got {policy.get('version')!r}")

    model_name = policy.get("model_name")
    if not isinstance(model_name, str) or not model_name.startswith("user.") or \
            not SAFE_ID_RE.match(model_name[len("user."):] or ""):
        _err(issues, "model_name", "must be a non-empty string starting with 'user.' using [A-Za-z0-9._-]")
    elif model_name == "user.MyHybridRouter":
        # Checks only the literal scaffold default — derived names like
        # user.Qwen3.5-2B-GGUF-Router can also collide across sessions, but
        # this offline validator sees one policy at a time and has no session
        # history to compare against. The collision risk is documented in
        # SKILL.md Step 2; enforcement requires the live server's /pull response.
        _warn(issues, "model_name",
              "using the bare scaffold default name 'user.MyHybridRouter' - if you register "
              "more than one router without renaming, later /pull calls will silently overwrite "
              "earlier ones (model_name is the collection identity). Give it a distinct name.")

    if policy.get("recipe") != "collection.router":
        _err(issues, "recipe", f"must be \"collection.router\", got {policy.get('recipe')!r}")

    components = policy.get("components")
    if not isinstance(components, list) or not components or not all(isinstance(c, str) and c for c in components):
        _err(issues, "components", "must be a non-empty array of non-empty strings")
        components = components if isinstance(components, list) else []

    routing = policy.get("routing")
    if not isinstance(routing, dict):
        _err(issues, "routing", "must be an object")
        return issues

    unknown_routing = set(routing.keys()) - ROUTING_KEYS
    if unknown_routing:
        _err(issues, "unknown_key", f"unknown routing key(s): {', '.join(sorted(unknown_routing))}")

    candidates = routing.get("candidates")
    if not isinstance(candidates, list) or not candidates or not all(isinstance(c, str) and c for c in candidates):
        _err(issues, "candidates", "routing.candidates must be a non-empty array of non-empty strings")
        candidates = candidates if isinstance(candidates, list) else []
    elif len(set(candidates)) != len(candidates):
        _err(issues, "candidates", "routing.candidates contains duplicates")

    default_model = routing.get("default_model")
    if default_model not in candidates:
        _err(issues, "default_model", f"'{default_model}' must be listed in routing.candidates")

    has_router = "router" in routing
    has_rules = "rules" in routing
    has_classifiers = "classifiers" in routing
    if has_router == has_rules:
        _err(issues, "mode", "routing must contain exactly one of 'router' or 'rules'")
    if has_router and has_classifiers:
        _err(issues, "mode", "'router' cannot be combined with 'classifiers'")
    if has_classifiers and not has_rules:
        _err(issues, "mode", "'classifiers' requires 'rules' (classifiers are referenced by rules)")

    needed_components = set(candidates)
    classifiers_by_id = {}

    if has_router:
        router = routing["router"]
        if not isinstance(router, dict):
            _err(issues, "router", "routing.router must be an object")
        else:
            unknown = set(router.keys()) - {"type", "model", "prompt"}
            if unknown:
                _err(issues, "router", f"unknown key(s): {', '.join(sorted(unknown))}")
            if router.get("type") != "llm":
                _err(issues, "router", "routing.router.type must be \"llm\"")
            model = router.get("model")
            if not isinstance(model, str) or not model:
                _err(issues, "router", "routing.router.model must be a non-empty string")
            else:
                needed_components.add(model)
            prompt = router.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                _err(issues, "router", "routing.router.prompt must be a non-empty string")
            elif BAD_ROUTER_PROMPT_RE.search(prompt):
                _warn(issues, "router_prompt_contract",
                      "prompt tells the model to reply with a bare model name, but the engine "
                      "always demands a JSON {model, rationale} reply and appends its own "
                      "instruction saying so - this line is redundant/misleading; author only "
                      "routing intent and let the engine own the reply format.")

    if has_classifiers:
        clfs = routing["classifiers"]
        if not isinstance(clfs, list):
            _err(issues, "classifiers", "routing.classifiers must be an array")
            clfs = []
        seen_ids = set()
        for i, clf in enumerate(clfs):
            p = f"routing.classifiers[{i}]"
            if not isinstance(clf, dict):
                _err(issues, "classifier", "must be an object", p)
                continue
            unknown_clf = set(clf.keys()) - CLASSIFIER_KEYS
            if unknown_clf:
                _err(issues, "unknown_key", f"unknown key(s): {', '.join(sorted(unknown_clf))}", p)

            cid = clf.get("id")
            if not isinstance(cid, str) or not cid:
                _err(issues, "classifier_id", "must be a non-empty string", p)
            elif cid in seen_ids:
                _err(issues, "classifier_id", f"duplicate id '{cid}'", p)
            else:
                seen_ids.add(cid)
                classifiers_by_id[cid] = clf

            ctype = clf.get("type")
            if ctype not in CLASSIFIER_TYPES:
                _err(issues, "classifier_type", f"type must be one of {sorted(CLASSIFIER_TYPES)}, got {ctype!r}", p)

            model = clf.get("model")
            if not isinstance(model, str) or not model:
                _err(issues, "classifier_model", "model must be a non-empty string", p)
            else:
                needed_components.add(model)

            on_error = clf.get("on_error")
            if on_error is not None and on_error not in ON_ERROR_VALUES:
                _err(issues, "on_error", f"must be one of {sorted(ON_ERROR_VALUES)}, got {on_error!r}", p)

            labels = clf.get("labels")
            if ctype == "llm":
                prompt = clf.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    _err(issues, "llm_classifier", "requires a non-empty 'prompt'", p)
                elif BAD_PROMPT_RE.search(prompt):
                    _warn(issues, "llm_classifier_prompt_contract",
                          "prompt tells the model to reply with a bare label, but the engine "
                          "appends its own JSON {model, rationale} contract and the classifier "
                          "scores the chosen label 1.0 — an authored reply-format instruction "
                          "causes weaker models to output a bare string the parser rejects, "
                          "making the rule silently never fire. Describe classification intent "
                          "only; do not instruct the model how to format its reply.", p)
                if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
                    _err(issues, "llm_classifier", "requires a non-empty 'labels' array", p)
            elif ctype == "semantic_similarity":
                if "labels" in clf:
                    _err(issues, "semantic_similarity",
                         "must not declare 'labels' - concept names in reference_phrases are the labels", p)
                rp = clf.get("reference_phrases")
                if not isinstance(rp, dict) or not rp:
                    _err(issues, "semantic_similarity", "requires a non-empty 'reference_phrases' object", p)
                    rp = {}
                else:
                    for concept, phrases in rp.items():
                        if not isinstance(phrases, list) or not phrases or \
                                not all(isinstance(x, str) and x for x in phrases):
                            _err(issues, "semantic_similarity",
                                 f"concept '{concept}' needs a non-empty array of non-empty phrases", p)
                labels = list(rp.keys())
            elif ctype == "classifier":
                if labels is not None and (not isinstance(labels, list) or not all(isinstance(x, str) and x for x in labels)):
                    _err(issues, "classifier", "'labels', if present, must be an array of non-empty strings", p)

            default_label = clf.get("default_label")
            if default_label is not None:
                if not isinstance(default_label, str) or not default_label:
                    _err(issues, "default_label", "must be a non-empty string", p)
                elif not labels:
                    _err(issues, "default_label", "set but classifier declares no labels/concepts to validate against", p)
                elif default_label not in labels:
                    _err(issues, "default_label", f"'{default_label}' is not in labels/concepts {labels}", p)

    if has_rules:
        rules = routing["rules"]
        if not isinstance(rules, list) or not rules:
            _err(issues, "rules", "routing.rules must be a non-empty array")
            rules = []
        seen_ids = set()
        for i, rule in enumerate(rules):
            p = f"routing.rules[{i}]"
            if not isinstance(rule, dict):
                _err(issues, "rule", "must be an object", p)
                continue
            unknown = set(rule.keys()) - {"id", "match", "route_to", "outputs"}
            if unknown:
                _err(issues, "rule", f"unknown key(s): {', '.join(sorted(unknown))}", p)

            rid = rule.get("id")
            if not isinstance(rid, str) or not rid or not SAFE_ID_RE.match(rid):
                _err(issues, "rule_id", "must be a non-empty string matching [A-Za-z0-9._-]", p)
            elif rid in seen_ids:
                _err(issues, "rule_id", f"duplicate id '{rid}'", p)
            else:
                seen_ids.add(rid)

            route_to = rule.get("route_to")
            if route_to not in candidates:
                _err(issues, "route_to", f"'{route_to}' must be listed in routing.candidates", p)

            if "outputs" in rule and not isinstance(rule["outputs"], dict):
                _err(issues, "outputs", "must be an object", p)

            match = rule.get("match")
            if not isinstance(match, dict) or not match:
                _err(issues, "match", "must be a non-empty object", p)
            else:
                validate_match_expr(match, classifiers_by_id, f"{p}.match", issues)

    missing = needed_components - set(components)
    if missing:
        _err(issues, "components", f"referenced but not declared in components: {sorted(missing)}")

    return issues


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", default="-", help="policy JSON file, or '-'/omitted for stdin")
    args = p.parse_args()

    raw = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
    try:
        policy = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"ready": False, "errors": [{"check": "json", "severity": "error", "message": str(e)}],
                          "warnings": [], "advisories": []}, indent=2))
        sys.exit(1)

    issues = validate(policy)
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    advisories = [i for i in issues if i["severity"] == "advisory"]
    result = {"ready": len(errors) == 0, "errors": errors, "warnings": warnings, "advisories": advisories}
    print(json.dumps(result, indent=2))
    sys.exit(0 if len(errors) == 0 else 1)


if __name__ == "__main__":
    main()
