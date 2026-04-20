"""Tests for CFActor and key generation utilities."""
import pytest
from circuitforge_core.activitypub.actor import (
    CFActor,
    generate_rsa_keypair,
    make_actor,
    load_actor_from_key_file,
)


@pytest.fixture(scope="module")
def keypair():
    return generate_rsa_keypair(bits=1024)  # small key for speed in tests


@pytest.fixture(scope="module")
def actor(keypair):
    priv, pub = keypair
    return make_actor(
        actor_id="https://kiwi.example.com/actors/kiwi",
        username="kiwi",
        display_name="Kiwi Pantry Bot",
        private_key_pem=priv,
        public_key_pem=pub,
        summary="Community recipe posts from Kiwi.",
    )


class TestGenerateRsaKeypair:
    def test_returns_two_strings(self, keypair):
        priv, pub = keypair
        assert isinstance(priv, str)
        assert isinstance(pub, str)

    def test_private_key_pem_header(self, keypair):
        priv, _ = keypair
        assert "BEGIN PRIVATE KEY" in priv

    def test_public_key_pem_header(self, keypair):
        _, pub = keypair
        assert "BEGIN PUBLIC KEY" in pub

    def test_keys_are_different(self, keypair):
        priv, pub = keypair
        assert priv != pub


class TestMakeActor:
    def test_actor_id_stored(self, actor):
        assert actor.actor_id == "https://kiwi.example.com/actors/kiwi"

    def test_inbox_derived_from_actor_id(self, actor):
        assert actor.inbox_url == "https://kiwi.example.com/actors/kiwi/inbox"

    def test_outbox_derived_from_actor_id(self, actor):
        assert actor.outbox_url == "https://kiwi.example.com/actors/kiwi/outbox"

    def test_username(self, actor):
        assert actor.username == "kiwi"

    def test_display_name(self, actor):
        assert actor.display_name == "Kiwi Pantry Bot"

    def test_summary_stored(self, actor):
        assert actor.summary == "Community recipe posts from Kiwi."

    def test_icon_url_defaults_none(self, keypair):
        priv, pub = keypair
        a = make_actor("https://x.com/a", "a", "A", priv, pub)
        assert a.icon_url is None

    def test_actor_is_frozen(self, actor):
        with pytest.raises((AttributeError, TypeError)):
            actor.username = "changed"  # type: ignore[misc]


class TestToApDict:
    def test_type_is_application(self, actor):
        d = actor.to_ap_dict()
        assert d["type"] == "Application"

    def test_id_matches_actor_id(self, actor):
        d = actor.to_ap_dict()
        assert d["id"] == actor.actor_id

    def test_preferred_username(self, actor):
        d = actor.to_ap_dict()
        assert d["preferredUsername"] == "kiwi"

    def test_public_key_present(self, actor):
        d = actor.to_ap_dict()
        assert "publicKey" in d
        assert d["publicKey"]["publicKeyPem"] == actor.public_key_pem

    def test_key_id_includes_main_key_fragment(self, actor):
        d = actor.to_ap_dict()
        assert d["publicKey"]["id"].endswith("#main-key")

    def test_private_key_not_in_dict(self, actor):
        d = actor.to_ap_dict()
        import json
        serialized = json.dumps(d)
        assert "PRIVATE KEY" not in serialized

    def test_context_includes_security(self, actor):
        d = actor.to_ap_dict()
        ctx = d["@context"]
        assert "https://w3id.org/security/v1" in ctx

    def test_summary_included_when_set(self, actor):
        d = actor.to_ap_dict()
        assert d["summary"] == actor.summary

    def test_summary_omitted_when_none(self, keypair):
        priv, pub = keypair
        a = make_actor("https://x.com/a", "a", "A", priv, pub)
        d = a.to_ap_dict()
        assert "summary" not in d

    def test_icon_included_when_set(self, keypair):
        priv, pub = keypair
        a = make_actor("https://x.com/a", "a", "A", priv, pub, icon_url="https://x.com/icon.png")
        d = a.to_ap_dict()
        assert d["icon"]["url"] == "https://x.com/icon.png"

    def test_icon_omitted_when_none(self, actor):
        d = actor.to_ap_dict()
        assert "icon" not in d


class TestLoadActorFromKeyFile:
    def test_loads_and_derives_public_key(self, keypair, tmp_path):
        priv, pub = keypair
        key_file = tmp_path / "actor.pem"
        key_file.write_text(priv)
        loaded = load_actor_from_key_file(
            actor_id="https://test.example/actors/x",
            username="x",
            display_name="X",
            private_key_path=str(key_file),
        )
        assert "BEGIN PUBLIC KEY" in loaded.public_key_pem
        assert loaded.actor_id == "https://test.example/actors/x"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_actor_from_key_file(
                actor_id="https://x.com/a",
                username="a",
                display_name="A",
                private_key_path=str(tmp_path / "missing.pem"),
            )
