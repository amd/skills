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

The vocabulary - allowed keys at every level, condition names, classifier
types and their per-type required/forbidden fields, value ranges - is read
from route_policy.schema.json next to this script, a verbatim copy of the
server's schema. Run scripts/sync_schema.py to refresh it when Lemonade ships
router changes; nothing here hardcodes a key list.

Usage:
    python3 scripts/validate.py router.json
    python3 scripts/validate.py < router.json
    cat router.json | python3 scripts/validate.py -
    python3 scripts/validate.py --schema other.schema.json router.json

Exits 0 if no errors (warnings/advisories may still be present), 1 otherwise.
JSON to stdout.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCHEMA_PATH = os.path.join(HERE, "route_policy.schema.json")
SCHEMA_SOURCE_PATH = os.path.join(HERE, "route_policy.schema.source.json")

LOGICAL_KEYS = {"all", "any", "not"}
CLASSIFIER_BAND_KEYS = {"label", "min_score", "max_score"}
RESERVED_CLASSIFIER_TYPES = {
    "pii_detection", "prompt_safety", "language_detection",
    "domain_classification", "complexity", "sentiment",
}
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# JSON Schema keywords _check_value enforces, and ones that carry no
# constraint. Anything else found on a property is reported as unenforced so a
# schema refresh that introduces a new constraint kind is visible, not silent.
ENFORCED_KEYWORDS = {
    "type", "const", "enum", "minimum", "maximum", "minLength", "pattern",
    "minItems", "uniqueItems", "items", "minProperties", "additionalProperties",
}
ANNOTATION_KEYWORDS = {"description", "default", "title", "$comment", "examples"}


def _type_rules(classifier_def):
    """Per-type field rules from the classifier's allOf if/then conditionals.

    Returns (by_type, gated, unparsed): by_type maps a type to the fields it
    requires, forbids, and requires non-empty; gated maps a field to the types
    allowed to carry it (e.g. on_truncation); unparsed counts conditionals of a
    shape this function doesn't understand.
    """
    by_type, gated, unparsed = {}, {}, 0
    for entry in classifier_def.get("allOf", []):
        cond, then = entry.get("if", {}), entry.get("then", {})
        type_const = cond.get("properties", {}).get("type", {}).get("const")
        if type_const:
            negated = then.get("not", {})
            forbidden = set(negated.get("required", []))
            for alt in negated.get("anyOf", []):
                forbidden |= set(alt.get("required", []))
            by_type[type_const] = {
                "required": set(then.get("required", [])),
                "forbidden": forbidden,
                "non_empty": {k for k, v in then.get("properties", {}).items()
                              if v.get("minItems", 0) >= 1},
            }
        elif cond.get("required") and "enum" in then.get("properties", {}).get("type", {}):
            for field in cond["required"]:
                gated[field] = set(then["properties"]["type"]["enum"])
        else:
            unparsed += 1
    return by_type, gated, unparsed


def _unenforced_keywords(prop_schemas):
    found = set()
    for prop in prop_schemas.values():
        stack = [prop]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            found |= set(node) - ENFORCED_KEYWORDS - ANNOTATION_KEYWORDS - {"$ref"}
            stack.extend(v for k, v in node.items() if k in ("items", "additionalProperties"))
    return found


def load_vocab(schema):
    """Derive every key set and enum the validator needs from the schema."""
    defs = schema["$defs"]
    clf = defs["classifier"]
    by_type, gated, unparsed = _type_rules(clf)
    all_types = set(clf["properties"]["type"]["enum"])
    leaf_props = defs["match_expr"]["properties"]
    return {
        "root_keys": set(schema["properties"]),
        "routing_keys": set(defs["routing"]["properties"]),
        "router_keys": set(defs["router_sugar"]["properties"]),
        "classifier_keys": set(clf["properties"]),
        "classifier_types": all_types - RESERVED_CLASSIFIER_TYPES,
        "reserved_classifier_types": all_types & RESERVED_CLASSIFIER_TYPES,
        "on_error_values": set(clf["properties"]["on_error"]["enum"]),
        "rule_keys": set(defs["rule"]["properties"]),
        "leaf_keys": set(leaf_props) - LOGICAL_KEYS,
        "metadata_keys": set(defs["metadata_match"]["properties"]),
        "type_gated_fields": set(gated),
        # Not sets - consumed by the checks below, skipped by sync_schema's diff.
        "_leaf_schemas": leaf_props,
        "_classifier_schemas": clf["properties"],
        "_type_rules": by_type,
        "_gated": gated,
        "_unparsed_conditionals": unparsed,
        "_unenforced_keywords": _unenforced_keywords(leaf_props) | _unenforced_keywords(clf["properties"]),
    }


def _load_schema(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# None only when the bundled schema is missing; main() reports that. Kept
# importable without it so sync_schema.py can bootstrap the first copy.
VOCAB = load_vocab(_load_schema(DEFAULT_SCHEMA_PATH)) if os.path.exists(DEFAULT_SCHEMA_PATH) else None

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


def _is_score(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0


_JSON_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _check_value(value, prop, name, issues, check, path):
    """Check one value against a property schema from the bundled schema.

    Returns True when it conforms. Covers the JSON Schema keywords the route
    policy schema uses on properties (see ENFORCED_KEYWORDS)."""
    types = prop.get("type")
    if types is not None:
        types = types if isinstance(types, list) else [types]
        if not any(_JSON_TYPES[t](value) for t in types if t in _JSON_TYPES):
            _err(issues, check, f"'{name}' must be {' or '.join(types)}, got {value!r}", path)
            return False
    if "const" in prop and value != prop["const"]:
        _err(issues, check, f"'{name}' must be {prop['const']!r}, got {value!r}", path)
        return False
    if "enum" in prop and value not in prop["enum"]:
        _err(issues, check, f"'{name}' must be one of {prop['enum']}, got {value!r}", path)
        return False
    ok = True
    if _JSON_TYPES["number"](value):
        if "minimum" in prop and value < prop["minimum"]:
            _err(issues, check, f"'{name}' must be >= {prop['minimum']}, got {value!r}", path)
            ok = False
        if "maximum" in prop and value > prop["maximum"]:
            _err(issues, check, f"'{name}' must be <= {prop['maximum']}, got {value!r}", path)
            ok = False
    if isinstance(value, str):
        if len(value) < prop.get("minLength", 0):
            _err(issues, check, f"'{name}' must be a non-empty string", path)
            ok = False
        elif "pattern" in prop and not re.search(prop["pattern"], value):
            _err(issues, check, f"'{name}' value {value!r} is blank or contains control characters "
                                f"(must match {prop['pattern']})", path)
            ok = False
    if isinstance(value, list):
        if len(value) < prop.get("minItems", 0):
            _err(issues, check, f"'{name}' must have at least {prop['minItems']} item(s)", path)
            ok = False
        if prop.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            _err(issues, check, f"'{name}' must not contain duplicates", path)
            ok = False
        if isinstance(prop.get("items"), dict):
            for i, item in enumerate(value):
                ok = _check_value(item, prop["items"], f"{name}[{i}]", issues, check, path) and ok
    if isinstance(value, dict):
        if len(value) < prop.get("minProperties", 0):
            _err(issues, check, f"'{name}' must have at least {prop['minProperties']} entr(y/ies)", path)
            ok = False
        if isinstance(prop.get("additionalProperties"), dict):
            for k, v in value.items():
                ok = _check_value(v, prop["additionalProperties"], f"{name}.{k}", issues, check, path) and ok
    return ok


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
    unknown = set(expr.keys()) - VOCAB["leaf_keys"]
    if unknown:
        _err(issues, "unknown_key",
             f"unknown condition key(s): {', '.join(sorted(unknown))} "
             f"(known: {', '.join(sorted(VOCAB['leaf_keys']))})", path)

    # Every condition the schema defines except classifier/metadata (which
    # need cross-references, below) is checked straight from its property
    # schema, so a condition added upstream is validated after a schema sync
    # with no code change here.
    conditions = [k for k in expr if k in VOCAB["leaf_keys"] and k not in CLASSIFIER_BAND_KEYS]
    for key in conditions:
        if key in ("classifier", "metadata"):
            continue
        if not _check_value(expr[key], VOCAB["_leaf_schemas"][key], key, issues, key, path):
            continue
        if key == "regex":
            v = expr["regex"]
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

    if "classifier" in expr:
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
        md = expr["metadata"]
        if not isinstance(md, dict) or not isinstance(md.get("key"), str) or not md.get("key"):
            _err(issues, "metadata", "requires a non-empty string 'key'", path)
        else:
            unknown_md = set(md.keys()) - VOCAB["metadata_keys"]
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

    if not conditions:
        _err(issues, "empty_leaf", "leaf has no recognized condition", path)


def validate(policy):
    issues = []

    if not isinstance(policy, dict):
        _err(issues, "root", "policy must be a JSON object")
        return issues

    unknown_root = set(policy.keys()) - VOCAB["root_keys"]
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

    unknown_routing = set(routing.keys()) - VOCAB["routing_keys"]
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
            unknown = set(router.keys()) - VOCAB["router_keys"]
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
            unknown_clf = set(clf.keys()) - VOCAB["classifier_keys"]
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
            if ctype in VOCAB["reserved_classifier_types"]:
                _err(issues, "classifier_type",
                     f"type '{ctype}' is reserved in the schema but not implemented by the server "
                     f"yet (it rejects it at /pull); use one of {sorted(VOCAB['classifier_types'])}", p)
            elif ctype not in VOCAB["classifier_types"]:
                _err(issues, "classifier_type",
                     f"type must be one of {sorted(VOCAB['classifier_types'])}, got {ctype!r}", p)
            else:
                rules_for_type = VOCAB["_type_rules"].get(ctype, {})
                for field in sorted(rules_for_type.get("required", ())):
                    if field not in clf:
                        _err(issues, "classifier_fields", f"type '{ctype}' requires '{field}'", p)
                for field in sorted(rules_for_type.get("forbidden", ()) & clf.keys()):
                    hint = (" - concept names in reference_phrases are the labels"
                            if (ctype, field) == ("semantic_similarity", "labels") else "")
                    _err(issues, "classifier_fields", f"type '{ctype}' must not declare '{field}'{hint}", p)
                for field in sorted(rules_for_type.get("non_empty", ()) & clf.keys()):
                    if isinstance(clf[field], list) and not clf[field]:
                        _err(issues, "classifier_fields", f"type '{ctype}' requires a non-empty '{field}'", p)
                for field in sorted(VOCAB["type_gated_fields"] & clf.keys()):
                    if ctype not in VOCAB["_gated"][field]:
                        _err(issues, "classifier_fields",
                             f"'{field}' is only accepted on types {sorted(VOCAB['_gated'][field])}", p)

            # Value shapes (on_error enum, labels items, reference_phrases
            # structure, ...) straight from the schema's classifier properties.
            for field in sorted(set(clf.keys()) & VOCAB["classifier_keys"] - {"id", "type"}):
                _check_value(clf[field], VOCAB["_classifier_schemas"][field], field, issues, field, p)

            model = clf.get("model")
            if "model" in clf and (not isinstance(model, str) or not model):
                _err(issues, "classifier_model", "model must be a non-empty string", p)
            elif model:
                needed_components.add(model)

            labels = clf.get("labels")
            if ctype == "llm":
                prompt = clf.get("prompt")
                if "prompt" in clf and (not isinstance(prompt, str) or not prompt.strip()):
                    _err(issues, "llm_classifier", "requires a non-empty 'prompt'", p)
                elif isinstance(prompt, str) and BAD_PROMPT_RE.search(prompt):
                    _warn(issues, "llm_classifier_prompt_contract",
                          "prompt tells the model to reply with a bare label, but the engine "
                          "appends its own JSON {model, rationale} contract and the classifier "
                          "scores the chosen label 1.0 — an authored reply-format instruction "
                          "causes weaker models to output a bare string the parser rejects, "
                          "making the rule silently never fire. Describe classification intent "
                          "only; do not instruct the model how to format its reply.", p)
            if ctype == "semantic_similarity":
                rp = clf.get("reference_phrases")
                labels = list(rp.keys()) if isinstance(rp, dict) else []
                # Stricter than the schema (its phrase items carry no
                # minLength): parse_reference_phrases rejects both at /pull.
                for concept, phrases in (rp.items() if isinstance(rp, dict) else ()):
                    if not concept:
                        _err(issues, "reference_phrases", "concept names must be non-empty", p)
                    if isinstance(phrases, list) and any(x == "" for x in phrases):
                        _err(issues, "reference_phrases",
                             f"concept '{concept}' has an empty phrase; phrases must be non-empty strings", p)
            elif not isinstance(labels, list):
                labels = []

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
            unknown = set(rule.keys()) - VOCAB["rule_keys"]
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
    p.add_argument("--schema", default=DEFAULT_SCHEMA_PATH,
                   help="route policy schema to validate against (default: the bundled copy)")
    args = p.parse_args()

    global VOCAB
    if not os.path.exists(args.schema):
        print(f"error: schema not found at {args.schema}; run scripts/sync_schema.py", file=sys.stderr)
        sys.exit(2)
    if args.schema != DEFAULT_SCHEMA_PATH:
        VOCAB = load_vocab(_load_schema(args.schema))
    schema_source = {"path": os.path.basename(args.schema)}
    if args.schema == DEFAULT_SCHEMA_PATH and os.path.exists(SCHEMA_SOURCE_PATH):
        with open(SCHEMA_SOURCE_PATH, encoding="utf-8") as f:
            schema_source.update(json.load(f))

    raw = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
    try:
        policy = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"ready": False, "errors": [{"check": "json", "severity": "error", "message": str(e)}],
                          "warnings": [], "advisories": [], "schema": schema_source}, indent=2))
        sys.exit(1)

    issues = validate(policy)
    # The schema grew a construct this validator can't read: say so rather than
    # pass the policy on a check that silently didn't run.
    if VOCAB["_unparsed_conditionals"] or VOCAB["_unenforced_keywords"]:
        _warn(issues, "schema_drift",
              "the schema uses constraints this validator does not enforce "
              f"({VOCAB['_unparsed_conditionals']} unrecognized classifier conditional(s); "
              f"keywords: {sorted(VOCAB['_unenforced_keywords']) or 'none'}) - confirm the policy "
              "with POST /api/v1/routing/validate on the target server")
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    advisories = [i for i in issues if i["severity"] == "advisory"]
    result = {"ready": len(errors) == 0, "errors": errors, "warnings": warnings, "advisories": advisories,
              "schema": schema_source}
    print(json.dumps(result, indent=2))
    sys.exit(0 if len(errors) == 0 else 1)


if __name__ == "__main__":
    main()
