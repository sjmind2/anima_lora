from __future__ import annotations

from typing import Any

from workflow.i18n.backend import _load_locale, _resolve


_TRANSLATABLE_FIELD_KEYS = ("label", "description", "help")


def _translate_value(locale_msgs: dict, en_msgs: dict, *key_parts: str) -> str | None:
    val = _resolve(locale_msgs, list(key_parts))
    if val is not None:
        return val
    val = _resolve(en_msgs, list(key_parts))
    return val


def translate_schema(schema: dict, schema_name: str, locale: str) -> dict:
    locale_msgs = _load_locale(locale)
    en_msgs = _load_locale("en")
    schema_section = _resolve(locale_msgs, ["schema", schema_name])
    en_schema = _resolve(en_msgs, ["schema", schema_name])
    if not schema_section and not en_schema:
        return schema

    result = dict(schema)

    for fk in _TRANSLATABLE_FIELD_KEYS:
        val = _translate_value(locale_msgs, en_msgs, "schema", schema_name, "root", fk)
        if val is not None:
            result[fk] = val

    for group in result.get("groups", []):
        group_name = group.get("name", "")
        group_label = _translate_value(locale_msgs, en_msgs, "schema", schema_name, "group", group_name)
        if group_label is not None:
            group["label"] = group_label

        for field in group.get("fields", []):
            field_key = field.get("key", "")
            for fk in _TRANSLATABLE_FIELD_KEYS:
                val = _translate_value(locale_msgs, en_msgs, "schema", schema_name, fk, field_key)
                if val is not None:
                    field[fk] = val

            if "choice_labels" in field and isinstance(field["choice_labels"], dict):
                cl_section = _resolve(locale_msgs, ["schema", schema_name, "choice_labels", field_key])
                en_cl_section = _resolve(en_msgs, ["schema", schema_name, "choice_labels", field_key])
                for choice_key in list(field["choice_labels"].keys()):
                    if cl_section and isinstance(cl_section, dict) and choice_key in cl_section:
                        field["choice_labels"][choice_key] = cl_section[choice_key]
                    elif en_cl_section and isinstance(en_cl_section, dict) and choice_key in en_cl_section:
                        field["choice_labels"][choice_key] = en_cl_section[choice_key]

        if "combo_switches" in group and isinstance(group["combo_switches"], list):
            for cs in group["combo_switches"]:
                cs_key = cs.get("key", "")
                for fk in ("label", "description"):
                    val = _translate_value(locale_msgs, en_msgs, "schema", schema_name, "combo_switch", cs_key, fk)
                    if val is not None:
                        cs[fk] = val

    return result
