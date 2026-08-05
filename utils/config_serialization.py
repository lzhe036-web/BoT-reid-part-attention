# encoding: utf-8
"""Strict, auditable serialization for resolved YACS configuration evidence."""

from __future__ import absolute_import

import yaml
from yacs.config import CfgNode


BOT_CFG_TYPE_TAG = "__bot_cfg_type__"
BOT_CFG_VALUE_KEY = "value"
BOT_CFG_ITEMS_KEY = "items"
_SCALAR_TYPES = (bool, int, float, type(None))


def cfg_node_to_plain_mapping(value):
    """Copy a CfgNode tree directly, preserving every in-memory Python type."""
    if isinstance(value, CfgNode) or isinstance(value, dict):
        if BOT_CFG_TYPE_TAG in value:
            raise ValueError(
                "Configuration key {} is reserved for resolved evidence"
                .format(BOT_CFG_TYPE_TAG)
            )
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("Configuration mapping keys must be strings")
            result[key] = cfg_node_to_plain_mapping(item)
        return result
    if isinstance(value, list):
        return [cfg_node_to_plain_mapping(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cfg_node_to_plain_mapping(item) for item in value)
    if type(value) is str or type(value) in _SCALAR_TYPES:
        return value
    raise TypeError(
        "Unsupported resolved configuration value type: {}".format(
            type(value).__name__
        )
    )


def _encode_typed_value(value):
    if isinstance(value, dict):
        return {key: _encode_typed_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_typed_value(item) for item in value]
    if isinstance(value, tuple):
        return {
            BOT_CFG_TYPE_TAG: "tuple",
            BOT_CFG_ITEMS_KEY: [_encode_typed_value(item) for item in value],
        }
    if type(value) is str:
        return {
            BOT_CFG_TYPE_TAG: "str",
            BOT_CFG_VALUE_KEY: value,
        }
    if type(value) in _SCALAR_TYPES:
        return value
    raise TypeError(
        "Unsupported resolved configuration value type: {}".format(
            type(value).__name__
        )
    )


def _decode_typed_value(value, path="<root>"):
    if isinstance(value, dict):
        if BOT_CFG_TYPE_TAG in value:
            tag = value[BOT_CFG_TYPE_TAG]
            if type(tag) is not str:
                raise ValueError("{} has a non-string type tag".format(path))
            if tag == "str":
                expected_keys = {BOT_CFG_TYPE_TAG, BOT_CFG_VALUE_KEY}
                if set(value) != expected_keys:
                    raise ValueError(
                        "{} string wrapper must contain exactly {}".format(
                            path, sorted(expected_keys)
                        )
                    )
                restored = value[BOT_CFG_VALUE_KEY]
                if type(restored) is not str:
                    raise ValueError(
                        "{} string wrapper value must be a string".format(path)
                    )
                return restored
            if tag == "tuple":
                expected_keys = {BOT_CFG_TYPE_TAG, BOT_CFG_ITEMS_KEY}
                if set(value) != expected_keys:
                    raise ValueError(
                        "{} tuple wrapper must contain exactly {}".format(
                            path, sorted(expected_keys)
                        )
                    )
                items = value[BOT_CFG_ITEMS_KEY]
                if not isinstance(items, list):
                    raise ValueError(
                        "{} tuple wrapper items must be a list".format(path)
                    )
                return tuple(
                    _decode_typed_value(item, "{}[{}]".format(path, index))
                    for index, item in enumerate(items)
                )
            raise ValueError(
                "{} has unknown resolved configuration type tag {!r}".format(
                    path, tag
                )
            )
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("{} has a non-string mapping key".format(path))
            result[key] = _decode_typed_value(
                item, "{}.{}".format(path, key)
            )
        return result
    if isinstance(value, list):
        return [
            _decode_typed_value(item, "{}[{}]".format(path, index))
            for index, item in enumerate(value)
        ]
    if type(value) is str:
        raise ValueError("{} contains an unprotected string leaf".format(path))
    if type(value) in _SCALAR_TYPES:
        return value
    raise ValueError(
        "{} contains unsupported YAML type {}".format(path, type(value).__name__)
    )


def serialize_cfg_node_yaml(cfg_node):
    """Serialize a CfgNode using explicit string and tuple type wrappers."""
    plain = cfg_node_to_plain_mapping(cfg_node)
    if not isinstance(plain, dict):
        raise TypeError("Resolved configuration root must be a mapping")
    encoded = _encode_typed_value(plain)
    return yaml.safe_dump(encoded, default_flow_style=False, sort_keys=False)


def deserialize_cfg_node_yaml(value):
    """Strictly restore a mapping emitted by serialize_cfg_node_yaml."""
    try:
        payload = yaml.safe_load(value)
    except yaml.YAMLError as error:
        raise ValueError("Invalid resolved configuration YAML") from error
    if not isinstance(payload, dict):
        raise ValueError("Resolved configuration YAML root must be a mapping")
    restored = _decode_typed_value(payload)
    if not isinstance(restored, dict):
        raise ValueError("Restored resolved configuration must be a mapping")
    return restored
