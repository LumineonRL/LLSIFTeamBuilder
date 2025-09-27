use simulator_rs::accessory::factory::AccessoryFactory;
use simulator_rs::accessory::manager::AccessoryManager;
use std::collections::HashSet;
use std::path::Path;

fn create_factory() -> AccessoryFactory {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let base_path = Path::new(manifest_dir)
        .parent()
        .unwrap()
        .join("data")
        .join("accessories.json");
    AccessoryFactory::new(base_path.to_str().unwrap()).unwrap()
}

#[test]
fn test_add_accessory() {
    let factory = create_factory();
    let mut manager = AccessoryManager::new(factory);
    let manager_id = manager.add_accessory(1, 1).unwrap();
    assert_eq!(manager_id, 1);
    assert!(manager.get_accessory(1).is_some());
}

#[test]
fn test_remove_accessory() {
    let factory = create_factory();
    let mut manager = AccessoryManager::new(factory);
    manager.add_accessory(1, 1);
    assert!(manager.remove_accessory(1));
    assert!(manager.get_accessory(1).is_none());
}

#[test]
fn test_modify_accessory() {
    let factory = create_factory();
    let mut manager = AccessoryManager::new(factory);
    let manager_id = manager.add_accessory(1, 1).unwrap();
    assert!(manager.modify_accessory(manager_id, Some(5)));
    let accessory = manager.get_accessory(manager_id).unwrap();
    assert_eq!(accessory.skill_level, 5);
}

#[test]
fn test_get_unassigned_accessories() {
    let factory = create_factory();
    let mut manager = AccessoryManager::new(factory);
    let id1 = manager.add_accessory(1, 1).unwrap();
    let id2 = manager.add_accessory(2, 1).unwrap();
    let mut assigned_ids = HashSet::new();
    assigned_ids.insert(id1);
    let unassigned = manager.get_unassigned_accessories(&assigned_ids);
    assert_eq!(unassigned.len(), 1);
    assert_eq!(unassigned[0].manager_internal_id, id2);
}

#[test]
fn test_save_and_load() {
    let factory = create_factory();
    let mut manager = AccessoryManager::new(factory.clone());
    manager.add_accessory(1, 2);
    manager.add_accessory(3, 4);

    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("accessories.json");
    let file_path_str = file_path.to_str().unwrap();

    manager.save(file_path_str).unwrap();

    let mut new_manager = AccessoryManager::new(factory);
    new_manager.load(file_path_str).unwrap();

    assert_eq!(manager.accessories().len(), new_manager.accessories().len());
    let acc1 = manager.get_accessory(1).unwrap();
    let new_acc1 = new_manager.get_accessory(1).unwrap();
    assert_eq!(acc1, new_acc1);
}

#[test]
fn test_golden_master_loading() {
    let factory = create_factory();
    let mut manager = AccessoryManager::new(factory);

    // This JSON is generated from the Python implementation
    let golden_json_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("resources")
        .join("golden_accessory.json");

    // To regenerate the golden file, run:
    // python3 -m simulator.accessory.accessory_manager > simulator_rs/tests/resources/golden_accessory.json
    // For this to work, you need to add a main block to accessory_manager.py
    // that creates a manager, adds accessories, and prints manager.to_dict() as json

    manager.load(golden_json_path.to_str().unwrap()).unwrap();

    assert_eq!(manager.accessories().len(), 2);

    let acc1 = manager.get_player_accessory(1).unwrap();
    assert_eq!(acc1.accessory.data.accessory_id, 1);
    assert_eq!(acc1.accessory.skill_level, 2);

    let acc2 = manager.get_player_accessory(2).unwrap();
    assert_eq!(acc2.accessory.data.accessory_id, 3);
    assert_eq!(acc2.accessory.skill_level, 4);
}
