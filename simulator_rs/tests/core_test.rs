use simulator_rs::core::{LeaderSkill, Number, Skill};

#[test]
fn test_skill_deserialization() {
    let skill_json = std::fs::read_to_string("tests/data/skill.json").unwrap();
    let skill: Skill = serde_json::from_str(&skill_json).unwrap();

    assert_eq!(skill.skill_type.as_deref(), Some("scorer"));
    assert_eq!(skill.activation.as_deref(), Some("rhythm icons"));
    assert_eq!(skill.target.as_deref(), Some("all"));
    assert_eq!(
        skill.levels,
        vec![
            Number::Int(1),
            Number::Int(2),
            Number::Int(3),
            Number::Int(4),
            Number::Int(5),
            Number::Int(6),
            Number::Int(7),
            Number::Int(8)
        ]
    );
    assert_eq!(
        skill.thresholds,
        vec![
            Some(Number::Int(20)),
            Some(Number::Int(25)),
            Some(Number::Int(30)),
            Some(Number::Int(35)),
            Some(Number::Int(40)),
            Some(Number::Int(45)),
            Some(Number::Int(50)),
            Some(Number::Int(55))
        ]
    );
    assert_eq!(
        skill.chances,
        vec![0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51]
    );
    assert_eq!(
        skill.values,
        vec![
            Some(Number::Int(100)),
            Some(Number::Int(200)),
            Some(Number::Int(300)),
            Some(Number::Int(400)),
            Some(Number::Int(500)),
            Some(Number::Int(600)),
            Some(Number::Int(700)),
            Some(Number::Int(800))
        ]
    );
    assert_eq!(
        skill.durations,
        vec![
            Some(Number::Float(3.0)),
            Some(Number::Float(3.5)),
            Some(Number::Float(4.0)),
            Some(Number::Float(4.5)),
            Some(Number::Float(5.0)),
            Some(Number::Float(5.5)),
            Some(Number::Float(6.0)),
            Some(Number::Float(6.5))
        ]
    );
}

#[test]
fn test_leader_skill_deserialization() {
    let leader_skill_json = std::fs::read_to_string("tests/data/leader_skill.json").unwrap();
    let leader_skill: LeaderSkill = serde_json::from_str(&leader_skill_json).unwrap();

    assert_eq!(leader_skill.attribute.as_deref(), Some("smile"));
    assert_eq!(leader_skill.secondary_attribute, None);
    assert_eq!(leader_skill.value, Some(0.09));
    assert_eq!(leader_skill.extra_attribute(), Some(&"cool".to_string()));
    assert_eq!(leader_skill.extra_target(), Some(&"bibi".to_string()));
    assert_eq!(leader_skill.extra_value(), 0.03);
}
