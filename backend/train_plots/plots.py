import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- CONFIG ---
LOG_FILE = r"E:\IRP\plots\train.txt"  # path to your log
START_EPOCH = 1          # first epoch number
ticks_interval = 5  # show every 5th epoch tick; tweak as needed

# prepare containers
epochs    = []
train_tot = []
val_tot   = []
train_ctc = []
val_ctc   = []
train_err = []
val_err   = []

# regexes
epoch_pat = re.compile(r"===== Epoch\s+(\d+)\s+=====")
train_pat = re.compile(r"Training: Total Loss: ([0-9.]+), CTC Loss: ([0-9.]+), ErrorCls Loss: ([0-9.]+)")
val_pat   = re.compile(r"Validation: Total Loss: ([0-9.]+), CTC Loss: ([0-9.]+), ErrorCls Loss: ([0-9.]+)")

current_epoch = None
pending_train = None

with open(LOG_FILE, "r") as f:
    for line in f:
        line = line.strip()
        # new epoch header?
        m = epoch_pat.search(line)
        if m:
            current_epoch = int(m.group(1))
            pending_train = None
            continue

        # training line?
        m = train_pat.search(line)
        if m and current_epoch is not None:
            pending_train = (
                float(m.group(1)),
                float(m.group(2)),
                float(m.group(3))
            )
            continue

        # validation line?
        m = val_pat.search(line)
        if m and current_epoch is not None and pending_train is not None:
            # now we have a full epoch’s worth of data → record it
            epochs.append(current_epoch)
            train_tot.append(pending_train[0])
            train_ctc.append(pending_train[1])
            train_err.append(pending_train[2])
            val_tot.append( float(m.group(1)) )
            val_ctc.append( float(m.group(2)) )
            val_err.append( float(m.group(3)) )
            # reset for next epoch
            current_epoch = None
            pending_train = None

# sanity check
assert len(epochs) == len(train_tot) == len(val_tot), \
       f"Still mismatched: epochs={len(epochs)} train.txt={len(train_tot)} val={len(val_tot)}"



def save_plot(x, ys, labels, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(10,6), dpi=120)
    for y, lbl in zip(ys, labels):
        ax.plot(x, y, marker='o', label=lbl)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    # force integer ticks and skip some for readability
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    # hide ticks that aren’t multiples of ticks_interval
    # hide ticks that aren’t multiples of ticks_interval
    for label in ax.get_xticklabels():
        txt = label.get_text().strip()
        # convert any Unicode minus to ASCII hyphen
        txt = txt.replace('\u2212', '-')
        try:
            val = int(txt)
        except ValueError:
            # skip non‑integer ticks (like empty strings or “−6” with weird chars)
            continue
        if val % ticks_interval != 0:
            label.set_visible(False)


    ax.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# make the three plots
save_plot(epochs, [train_tot, val_tot],
          ["Train Total", "Val Total"],
          "Total Loss", "Total Loss per Epoch", r"E:\IRP\plots\total_loss.png")

save_plot(epochs, [train_ctc, val_ctc],
          ["Train CTC", "Val CTC"],
          "CTC Loss", "CTC Loss per Epoch", r"E:\IRP\plots\ctc_loss.png")

save_plot(epochs, [train_err, val_err],
          ["Train ErrorCls", "Val ErrorCls"],
          "ErrorCls Loss", "Error Classification Loss per Epoch", r"E:\IRP\plots\errorcls_loss.png")

print("All plots generated.")
