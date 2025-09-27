use simulator_rs::accessory::factory::AccessoryFactory;
use simulator_rs::accessory::manager::AccessoryManager;
use simulator_rs::card::card_factory::CardFactory;
use simulator_rs::card::deck::Deck;
use simulator_rs::sis::factory::SisFactory;
use simulator_rs::sis::manager::SisManager;
use std::path::PathBuf;
use std::sync::Arc;

fn main() {
    // --- Path Setup ---
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_path = manifest_dir
        .parent()
        .expect("Failed to get parent directory of the crate manifest.")
        .join("data");

    let cards_file = data_path.join("cards.json");
    let level_caps_file = data_path.join("level_caps.json");
    let level_cap_bonuses_file = data_path.join("level_cap_bonuses.json");
    let accessories_file = data_path.join("accessories.json");
    let sis_file = data_path.join("sis.json");

    let card_factory = CardFactory::new(
        cards_file
            .to_str()
            .expect("cards.json path is not valid UTF-8"),
        level_caps_file
            .to_str()
            .expect("level_caps.json path is not valid UTF-8"),
        level_cap_bonuses_file
            .to_str()
            .expect("level_cap_bonuses.json path is not valid UTF-8"),
    )
    .expect("Failed to create Card Factory.");

    let factory_arc = Arc::new(card_factory);
    let mut deck = Deck::new(Arc::clone(&factory_arc));

    println!("--- Deck and Card Example ---");
    if let Some(deck_id) = deck.add_card(1, true, 4, Some(100), Some(4)) {
        println!("Successfully added card with Deck ID: {deck_id}");
    }
    if let Some(card) = deck.get_card(1) {
        println!("{card}\n");
    }

    println!("--- Accessory Manager Example ---");

    let accessory_factory = AccessoryFactory::new(
        accessories_file
            .to_str()
            .expect("accessories.json path is not valid UTF-8"),
    )
    .expect("Failed to create Accessory Factory.");

    let mut acc_manager = AccessoryManager::new(accessory_factory);
    println!("Initialized a new, empty Accessory Manager.\n");

    println!("-> Adding accessories...");
    if let Some(id) = acc_manager.add_accessory(27, 4) {
        println!("Added accessory, received Manager ID: {id}");
    }
    if let Some(id) = acc_manager.add_accessory(35, 8) {
        println!("Added accessory, received Manager ID: {id}");
    }
    println!("\nCurrent Manager State:\n{acc_manager}\n");

    println!("-> Modifying accessory with Manager ID 1 to Skill Level 5...");
    if acc_manager.modify_accessory(1, Some(5)) {
        println!("Modification successful.");
    }
    println!("\nCurrent Manager State:\n{acc_manager}\n");

    let acc_save_path = manifest_dir.join("temp").join("player_accessories.json");
    println!("-> Saving manager state to: {acc_save_path:?}");
    if let Err(e) = acc_manager.save(acc_save_path.to_str().unwrap()) {
        eprintln!("Error saving state: {e}");
    } else {
        println!("Save successful.");
    }

    acc_manager.remove_accessory(2);
    println!("\n-> Removed accessory 2. Current state:\n{acc_manager}\n");

    println!("-> Loading manager state from file...");
    if let Err(e) = acc_manager.load(acc_save_path.to_str().unwrap()) {
        eprintln!("Error loading state: {e}");
    } else {
        println!("Load successful.");
    }
    println!("\nFinal Manager State (Restored from file):\n{acc_manager}");
    println!("\n----------------------------------------\n");

    println!("--- SIS Manager Example ---");

    let sis_factory = SisFactory::new(sis_file.to_str().expect("sis.json path is not valid UTF-8"))
        .expect("Failed to create SIS Factory.");

    let mut sis_manager = SisManager::new(sis_factory);
    println!("Initialized a new, empty SIS Manager.\n");

    println!("-> Adding SIS...");
    if let Some(id) = sis_manager.add_sis(2) {
        println!("Added SIS, received Manager ID: {id}");
    }
    if let Some(id) = sis_manager.add_sis(100) {
        println!("Added SIS, received Manager ID: {id}");
    }
    println!("\nCurrent Manager State:\n{sis_manager}\n");

    let sis_save_path = manifest_dir.join("temp").join("player_sis.json");
    println!("-> Saving manager state to: {sis_save_path:?}");
    if let Err(e) = sis_manager.save(sis_save_path.to_str().unwrap()) {
        eprintln!("Error saving state: {e}");
    } else {
        println!("Save successful.");
    }

    sis_manager.remove_sis(1);
    println!("\n-> Removed SIS with Manager ID 1. Current state:\n{sis_manager}\n");

    println!("-> Loading manager state from file...");
    if let Err(e) = sis_manager.load(sis_save_path.to_str().unwrap()) {
        eprintln!("Error loading state: {e}");
    } else {
        println!("Load successful.");
    }
    println!("\nFinal Manager State (Restored from file):\n{sis_manager}");
}
