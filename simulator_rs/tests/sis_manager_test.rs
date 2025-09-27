use simulator_rs::sis::factory::SisFactory;
use simulator_rs::sis::manager::SisManager;
use std::collections::HashSet;
use std::path::Path;

fn create_factory() -> SisFactory {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let base_path = Path::new(manifest_dir)
        .parent()
        .unwrap()
        .join("data")
        .join("sis.json");
    SisFactory::new(base_path.to_str().unwrap()).unwrap()
}

#[test]
fn test_add_sis() {
    let factory = create_factory();
    let mut manager = SisManager::new(factory);
    let manager_id = manager.add_sis(1).unwrap();
    assert_eq!(manager_id, 1);
    assert!(manager.get_sis(1).is_some());
}

#[test]
fn test_remove_sis() {
    let factory = create_factory();
    let mut manager = SisManager::new(factory);
    manager.add_sis(1);
    assert!(manager.remove_sis(1));
    assert!(manager.get_sis(1).is_none());
}

#[test]
fn test_get_unassigned_sis() {
    let factory = create_factory();
    let mut manager = SisManager::new(factory);
    let id1 = manager.add_sis(1).unwrap();
    let id2 = manager.add_sis(2).unwrap();
    let mut assigned_ids = HashSet::new();
    assigned_ids.insert(id1);
    let unassigned = manager.get_unassigned_sis(&assigned_ids);
    assert_eq!(unassigned.len(), 1);
    assert_eq!(unassigned[0].manager_internal_id, id2);
}

#[test]
fn test_save_and_load() {
    let factory = create_factory();
    let mut manager = SisManager::new(factory.clone());
    manager.add_sis(1);
    manager.add_sis(3);

    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("sis.json");
    let file_path_str = file_path.to_str().unwrap();

    manager.save(file_path_str).unwrap();

    let mut new_manager = SisManager::new(factory);
    new_manager.load(file_path_str).unwrap();

    assert_eq!(manager.skills().len(), new_manager.skills().len());
    let sis1 = manager.get_sis(1).unwrap();
    let new_sis1 = new_manager.get_sis(1).unwrap();
    assert_eq!(sis1, new_sis1);
}

#[test]
fn test_golden_master_loading() {
    let factory = create_factory();
    let mut manager = SisManager::new(factory);

    let golden_json_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("resources")
        .join("golden_sis.json");

    manager.load(golden_json_path.to_str().unwrap()).unwrap();

    assert_eq!(manager.skills().len(), 2);

    let sis1 = manager.get_player_sis(1).unwrap();
    assert_eq!(sis1.sis.id(), 1);

    let sis2 = manager.get_player_sis(2).unwrap();
    assert_eq!(sis2.sis.id(), 2);
}
