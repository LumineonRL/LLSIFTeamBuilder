use simulator_rs::accessory::factory::AccessoryFactory;
use simulator_rs::accessory::stats::AccessoryStats;
use simulator_rs::core::skill::Number;
use std::path::PathBuf;

fn get_accessories_json_path() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../data/accessories.json");
    dunce::canonicalize(&path)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

#[test]
fn test_create_accessory_level_1() {
    let factory = AccessoryFactory::new(&get_accessories_json_path()).unwrap();
    let accessory = factory.create_accessory(1, 1).unwrap();

    assert_eq!(accessory.data.accessory_id, 1);
    assert_eq!(accessory.skill_level, 1);

    // Expected stats for level 1
    let expected_stats = AccessoryStats::new(30, 90, 50);
    assert_eq!(accessory.stats, expected_stats);

    // Expected skill values for level 1
    assert_eq!(accessory.skill_chance(), Some(25.0));
    assert_eq!(accessory.skill_threshold(), Some(0));
    assert_eq!(accessory.skill_duration(), Some(3.3));
    assert_eq!(accessory.skill_value(), Some(Number::Int(20)));
}

#[test]
fn test_create_accessory_level_8() {
    let factory = AccessoryFactory::new(&get_accessories_json_path()).unwrap();
    let accessory = factory.create_accessory(1, 8).unwrap();

    assert_eq!(accessory.data.accessory_id, 1);
    assert_eq!(accessory.skill_level, 8);

    // Expected stats for level 8
    let expected_stats = AccessoryStats::new(390, 970, 590);
    assert_eq!(accessory.stats, expected_stats);

    // Expected skill values for level 8
    assert_eq!(accessory.skill_chance(), Some(46.0));
    assert_eq!(accessory.skill_threshold(), Some(0));
    assert_eq!(accessory.skill_duration(), Some(5.0));
    assert_eq!(accessory.skill_value(), Some(Number::Int(34)));
}

#[test]
fn test_invalid_accessory_id() {
    let factory = AccessoryFactory::new(&get_accessories_json_path()).unwrap();
    let accessory = factory.create_accessory(999999, 1);
    assert!(accessory.is_none());
}

#[test]
fn test_invalid_skill_level() {
    let factory = AccessoryFactory::new(&get_accessories_json_path()).unwrap();
    let accessory_high = factory.create_accessory(1, 9);
    assert!(
        accessory_high.is_none(),
        "Accessory should not be created with skill level 9"
    );

    let accessory_low = factory.create_accessory(1, 0);
    assert!(
        accessory_low.is_none(),
        "Accessory should not be created with skill level 0"
    );
}
