use simulator_rs::sis::factory::SisFactory;

const SIS_JSON_PATH: &str = "../data/sis.json";

#[test]
fn test_create_sis_from_factory() {
    let factory = SisFactory::new(SIS_JSON_PATH).expect("Failed to create SIS factory");

    let sis = factory.create_sis(3).expect("SIS with ID 3 not found");

    assert_eq!(sis.id(), 3);
    assert_eq!(sis.name(), "Cool Kiss");
    assert_eq!(sis.effect(), "self flat boost");
    assert_eq!(sis.slots(), 1);
    assert_eq!(sis.attribute(), "Cool");
    assert_eq!(sis.group(), Some(""));
    assert_eq!(sis.equip_restriction(), Some(""));
    assert_eq!(sis.target(), Some("self"));
    assert_eq!(sis.value(), 200.0);
}

#[test]
fn test_create_non_existent_sis() {
    let factory = SisFactory::new(SIS_JSON_PATH).expect("Failed to create SIS factory");
    let sis = factory.create_sis(9999);
    assert!(sis.is_none());
}

#[test]
fn test_display_sis() {
    let factory = SisFactory::new(SIS_JSON_PATH).expect("Failed to create SIS factory");
    let sis = factory.create_sis(3).expect("SIS with ID 3 not found");
    let display_str = format!("{sis}");

    let expected = "<SIS id=3 name='Cool Kiss'>\n  - Effect: self flat boost (200)\n  - Slots: 1, Attribute: Cool\n";

    assert_eq!(display_str, expected);
}
