// =============================================================================
// Module: neurisc_soc
// Description: NeuroRISC SoC Top-Level Integration
// 
// Integrated Components:
// - Custom instruction decoder (control interface)
// - 8x8 Systolic Array NPU
// - Activation function unit
// - Weight buffer (256KB dual-port SRAM)
// - Activation buffer (128KB double-buffered SRAM)
// - DMA controller with 2D strided access
// - Performance counters
// - Memory-mapped control registers
//
// Memory Map:
// - 0x80001000-0x80001FFF: DMA control registers
// - 0x80002000-0x80002FFF: NPU control registers
// - 0x80010000-0x8001FFFF: Weight buffer access
// - 0x80020000-0x8002FFFF: Activation buffer access
// - 0x80030000-0x8003FFFF: Result buffer access
// =============================================================================

module neurisc_soc (
    input  logic        clock,              // System clock
    input  logic        reset,              // System reset (active high)
    
    // External memory-mapped register interface
    input  logic        reg_write_en,       // Register write enable
    input  logic        reg_read_en,        // Register read enable
    input  logic [31:0] reg_addr,           // Register address
    input  logic [31:0] reg_write_data,     // Register write data
    output logic [31:0] reg_read_data,      // Register read data
    output logic        reg_ready,          // Register access ready
    
    // External memory interface (for DMA access)
    output logic        ext_mem_read_en,    // External memory read enable
    output logic        ext_mem_write_en,   // External memory write enable
    output logic [31:0] ext_mem_addr,       // External memory address
    input  logic [63:0] ext_mem_read_data,  // External memory read data
    output logic [63:0] ext_mem_write_data, // External memory write data
    input  logic        ext_mem_ready,      // External memory ready
    
    // System status outputs
    output logic        npu_busy,           // NPU computation in progress
    output logic        npu_done,           // NPU computation complete
    output logic        dma_busy,           // DMA transfer in progress
    output logic [31:0] performance_cycles  // NPU performance cycle counter
);

    // ==========================================================================
    // Internal Signals - Custom Instruction Decoder
    // ==========================================================================
    logic        decoder_npu_start;
    logic        decoder_npu_clear;
    logic [1:0]  decoder_activation_select;
    logic        decoder_dma_start;
    logic [31:0] decoder_dma_src_addr;
    logic [31:0] decoder_dma_dst_addr;
    logic [15:0] decoder_dma_size;
    logic        decoder_dma_mode_2d;
    logic [15:0] decoder_dma_row_count;
    logic [15:0] decoder_dma_col_count;
    logic [15:0] decoder_dma_src_stride;
    logic [15:0] decoder_dma_dst_stride;
    logic        decoder_dma_busy;
    logic        decoder_dma_done;
    
    // ==========================================================================
    // Internal Signals - Systolic Array NPU
    // ==========================================================================
    logic signed [7:0]  npu_weight_in [0:7];
    logic signed [7:0]  npu_input_in [0:7];
    logic signed [19:0] npu_result [0:7][0:7];
    logic        npu_start;
    logic        npu_done_internal;
    logic [7:0]  npu_cycle_count;
    
    // ==========================================================================
    // Internal Signals - Weight Buffer
    // ==========================================================================
    logic        weight_buf_port_a_enable;
    logic        weight_buf_port_a_write_en;
    logic [14:0] weight_buf_port_a_addr;
    logic [63:0] weight_buf_port_a_write_data;
    logic [63:0] weight_buf_port_a_read_data;
    logic        weight_buf_port_b_enable;
    logic        weight_buf_port_b_write_en;
    logic [14:0] weight_buf_port_b_addr;
    logic [63:0] weight_buf_port_b_write_data;
    logic [63:0] weight_buf_port_b_read_data;
    
    // ==========================================================================
    // Internal Signals - Activation Buffer
    // ==========================================================================
    logic        act_buf_swap_banks;
    logic        act_buf_active_bank;
    logic        act_buf_read_enable;
    logic [12:0] act_buf_read_addr;
    logic [63:0] act_buf_read_data;
    logic        act_buf_write_enable;
    logic [12:0] act_buf_write_addr;
    logic [63:0] act_buf_write_data;
    
    // ==========================================================================
    // Internal Signals - Activation Unit
    // ==========================================================================
    logic signed [15:0] activation_data_in;
    logic signed [15:0] activation_data_out;
    
    // ==========================================================================
    // Internal Signals - DMA Controller
    // ==========================================================================
    logic        dma_start;
    logic        dma_busy_internal;
    logic        dma_done_internal;
    logic        dma_mem_read_en;
    logic        dma_mem_write_en;
    logic [31:0] dma_mem_read_addr;
    logic [31:0] dma_mem_write_addr;
    logic [63:0] dma_mem_read_data;
    logic [63:0] dma_mem_write_data;
    logic        dma_mem_ready;
    
    // ==========================================================================
    // Internal Signals - Performance Counter
    // ==========================================================================
    logic [31:0] npu_performance_counter;
    logic        npu_counter_enable;
    
    // ==========================================================================
    // Internal Signals - Data Path Connections
    // ==========================================================================
    logic [7:0]  weight_feeder_counter;
    logic [7:0]  input_feeder_counter;
    logic        data_feed_active;
    
    // ==========================================================================
    // Module Instantiation - Custom Instruction Decoder
    // ==========================================================================
    custom_instruction_decoder decoder_inst (
        .clock              (clock),
        .reset              (reset),
        .reg_write_en       (reg_write_en),
        .reg_read_en        (reg_read_en),
        .reg_addr           (reg_addr),
        .reg_write_data     (reg_write_data),
        .reg_read_data      (reg_read_data),
        .reg_ready          (reg_ready),
        .npu_start          (decoder_npu_start),
        .npu_clear          (decoder_npu_clear),
        .npu_done           (npu_done_internal),
        .npu_busy           (npu_busy),
        .npu_cycle_count    (npu_performance_counter),
        .dma_start          (decoder_dma_start),
        .dma_src_addr       (decoder_dma_src_addr),
        .dma_dst_addr       (decoder_dma_dst_addr),
        .dma_size           (decoder_dma_size),
        .dma_mode_2d        (decoder_dma_mode_2d),
        .dma_row_count      (decoder_dma_row_count),
        .dma_col_count      (decoder_dma_col_count),
        .dma_src_stride     (decoder_dma_src_stride),
        .dma_dst_stride     (decoder_dma_dst_stride),
        .dma_busy           (dma_busy_internal),
        .dma_done           (dma_done_internal),
        .activation_select  (decoder_activation_select)
    );
    
    // ==========================================================================
    // Module Instantiation - Systolic Array NPU
    // ==========================================================================
    systolic_array #(.SIZE(32)) npu_inst (
        .clock       (clock),
        .reset       (reset),
        .start       (npu_start),
        .accumulate  (1'b0),
        .weight_in   (npu_weight_in),
        .input_in    (npu_input_in),
        .result      (npu_result),
        .done        (npu_done_internal),
        .cycle_count (npu_cycle_count)
    );
    
    // ==========================================================================
    // Module Instantiation - Weight Buffer
    // ==========================================================================
    weight_buffer weight_buf_inst (
        .clock                 (clock),
        .reset                 (reset),
        .port_a_enable         (weight_buf_port_a_enable),
        .port_a_write_en       (weight_buf_port_a_write_en),
        .port_a_addr           (weight_buf_port_a_addr),
        .port_a_write_data     (weight_buf_port_a_write_data),
        .port_a_read_data      (weight_buf_port_a_read_data),
        .port_b_enable         (weight_buf_port_b_enable),
        .port_b_write_en       (weight_buf_port_b_write_en),
        .port_b_addr           (weight_buf_port_b_addr),
        .port_b_write_data     (weight_buf_port_b_write_data),
        .port_b_read_data      (weight_buf_port_b_read_data)
    );
    
    // ==========================================================================
    // Module Instantiation - Activation Buffer
    // ==========================================================================
    activation_buffer act_buf_inst (
        .clock         (clock),
        .reset         (reset),
        .swap_banks    (act_buf_swap_banks),
        .active_bank   (act_buf_active_bank),
        .read_enable   (act_buf_read_enable),
        .read_addr     (act_buf_read_addr),
        .read_data     (act_buf_read_data),
        .write_enable  (act_buf_write_enable),
        .write_addr    (act_buf_write_addr),
        .write_data    (act_buf_write_data)
    );
    
    // ==========================================================================
    // Module Instantiation - Activation Function Unit
    // ==========================================================================
    activation_unit activation_inst (
        .clock        (clock),
        .reset        (reset),
        .func_select  (decoder_activation_select),
        .data_in      (activation_data_in),
        .data_out     (activation_data_out)
    );
    
    // ==========================================================================
    // Module Instantiation - DMA Controller
    // ==========================================================================
    dma_controller dma_inst (
        .clock            (clock),
        .reset            (reset),
        .start            (dma_start),
        .busy             (dma_busy_internal),
        .done             (dma_done_internal),
        .src_base_addr    (decoder_dma_src_addr),
        .dst_base_addr    (decoder_dma_dst_addr),
        .transfer_size    (decoder_dma_size),
        .mode_2d          (decoder_dma_mode_2d),
        .row_count        (decoder_dma_row_count),
        .col_count        (decoder_dma_col_count),
        .src_row_stride   (decoder_dma_src_stride),
        .dst_row_stride   (decoder_dma_dst_stride),
        .mem_read_en      (dma_mem_read_en),
        .mem_write_en     (dma_mem_write_en),
        .mem_read_addr    (dma_mem_read_addr),
        .mem_write_addr   (dma_mem_write_addr),
        .mem_read_data    (dma_mem_read_data),
        .mem_write_data   (dma_mem_write_data),
        .mem_ready        (dma_mem_ready)
    );
    
    // ==========================================================================
    // NPU Data Feeder - Simplified Weight and Input Feeding Logic
    // ==========================================================================
    // In a real system, this would be controlled by a more sophisticated
    // sequencer that loads data from buffers and feeds the systolic array
    
    always_ff @(posedge clock) begin
        if (reset || decoder_npu_clear) begin
            weight_feeder_counter <= 8'h0;
            input_feeder_counter <= 8'h0;
            data_feed_active <= 1'b0;
        end else if (decoder_npu_start) begin
            data_feed_active <= 1'b1;
            weight_feeder_counter <= 8'h0;
            input_feeder_counter <= 8'h0;
        end else if (data_feed_active && (weight_feeder_counter < 8'd15)) begin
            weight_feeder_counter <= weight_feeder_counter + 1;
            input_feeder_counter <= input_feeder_counter + 1;
        end else if (npu_done_internal) begin
            data_feed_active <= 1'b0;
        end
    end
    
    // Feed weights and inputs from buffers to systolic array
    // This is a simplified version - real implementation would sequence properly
    always_comb begin
        for (int i = 0; i < 8; i++) begin
            // Extract 8-bit weights from 64-bit buffer read
            npu_weight_in[i] = weight_buf_port_a_read_data[i*8 +: 8];
            // Extract 8-bit inputs from 64-bit buffer read
            npu_input_in[i] = act_buf_read_data[i*8 +: 8];
        end
    end
    
    assign npu_start = decoder_npu_start;
    
    // ==========================================================================
    // Performance Counter - Tracks NPU Computation Cycles
    // ==========================================================================
    assign npu_counter_enable = npu_busy;
    
    always_ff @(posedge clock) begin
        if (reset || decoder_npu_clear) begin
            npu_performance_counter <= 32'h0;
        end else if (npu_counter_enable) begin
            npu_performance_counter <= npu_performance_counter + 1;
        end
    end
    
    // ==========================================================================
    // DMA-to-Memory Interface Connection
    // ==========================================================================
    assign dma_start = decoder_dma_start;
    assign dma_mem_ready = ext_mem_ready;
    assign dma_mem_read_data = ext_mem_read_data;
    
    assign ext_mem_read_en = dma_mem_read_en;
    assign ext_mem_write_en = dma_mem_write_en;
    assign ext_mem_addr = dma_mem_write_en ? dma_mem_write_addr : dma_mem_read_addr;
    assign ext_mem_write_data = dma_mem_write_data;
    
    // ==========================================================================
    // Buffer Access Control (simplified)
    // ==========================================================================
    // Port A: NPU reads weights
    assign weight_buf_port_a_enable = data_feed_active;
    assign weight_buf_port_a_write_en = 1'b0;  // NPU only reads
    assign weight_buf_port_a_addr = {7'b0, weight_feeder_counter};  // Zero-extend to 15 bits
    assign weight_buf_port_a_write_data = 64'h0;
    
    // Port B: DMA writes weights (simplified - would need proper arbitration)
    assign weight_buf_port_b_enable = 1'b0;  // Controlled by DMA in real system
    assign weight_buf_port_b_write_en = 1'b0;
    assign weight_buf_port_b_addr = 15'h0;
    assign weight_buf_port_b_write_data = 64'h0;
    
    // Activation buffer: NPU reads inputs
    assign act_buf_read_enable = data_feed_active;
    assign act_buf_read_addr = {5'b0, input_feeder_counter};  // Zero-extend to 13 bits
    assign act_buf_write_enable = 1'b0;  // Controlled by DMA in real system
    assign act_buf_write_addr = 13'h0;
    assign act_buf_write_data = 64'h0;
    assign act_buf_swap_banks = npu_done_internal;  // Swap on completion
    
    // Activation function: process NPU results
    // Simplified - would need proper result sequencing in real system
    assign activation_data_in = npu_result[0][0][15:0];  // Sample one result
    
    // ==========================================================================
    // System Status Outputs
    // ==========================================================================
    assign npu_busy = data_feed_active || (npu_cycle_count != 8'h0);
    assign npu_done = npu_done_internal;
    assign dma_busy = dma_busy_internal;
    assign performance_cycles = npu_performance_counter;

endmodule
