from standalonecad.planning import direct_plan


def test_creation_shortcuts_do_not_hijack_edit_requests():
    assert direct_plan('80×60×8 플레이트 길이를 100mm로 수정해줘', {}) is None
    assert direct_plan('모듈 2 잇수 25 치폭 20mm 보어 15mm 스퍼기어 길이를 수정해줘', {}) is None
    assert direct_plan('M12 육각볼트 전체 길이 50mm를 60mm로 바꿔줘', {}) is None


def test_existing_deterministic_creation_routes_are_preserved():
    plate = direct_plan('80×60×8 플레이트 만들어줘', {})
    assert plate and plate['calls'][-1]['tool'] == 'cad_create_box'

    gear = direct_plan('모듈 2 잇수 25 치폭 20mm 보어 15mm 스퍼기어 만들어줘', {})
    assert gear and gear['calls'][-1]['tool'] == 'cad_create_spur_gear'

    bolt = direct_plan('M12 육각볼트 전체 길이 50mm 나사산 길이 30mm 만들어줘', {})
    assert bolt and bolt['calls'][-1]['tool'] == 'cad_create_metric_hex_bolt'


def test_explicit_replace_current_still_uses_deterministic_creation():
    state={'document':{'type':'part','has_geometry':True},'selection':{}}
    plan=direct_plan('기존 모델 지우고 80×60×8 플레이트 만들어줘', state)
    assert plan and [c['tool'] for c in plan['calls']] == ['cad_create_box']
    assert plan['calls'][0]['arguments']['replace'] is True
